# mylego-vision - what can I build from my LEGO parts?
# scan parts with CV (Brickognize + Ollama vision), keep an inventory,
# rank every real LEGO set by buildability, ask a local LLM for MOC ideas.
# copyright (c) 2026 cocomelonc
# author: cocomelonc
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageChops, ImageFilter, ImageOps
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("mylego")

_configured_db = Path(os.getenv("LEGO_DB", "lego.db"))
DB_PATH = str(
    _configured_db if _configured_db.is_absolute() else BASE_DIR / _configured_db
)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.10.10.95:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3.6:27b")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen3.6:27b")
BRICKOGNIZE_URL = "https://api.brickognize.com/predict/"
PILE_MAX_PARTS = 20
PILE_WORKING_SIDE = 960
IDEA_PREVIEW_CACHE_LIMIT = 60
IDEA_PREVIEW_MAX_BYTES = 8 * 1024 * 1024
IDEA_PLACEHOLDER_URL = "/static/lego-placeholder.svg"
REAL_IMAGE_HOSTS = {"cdn.rebrickable.com"}

MIN_SET_PARTS = 5      # ignore 1-piece service packs
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
COLOR_POPULARITY_POWER = 0.25
SPECIAL_FINISH_MARKERS = (
    "chrome", "glitter", "metallic", "pearl", "satin", "speckle",
)


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def ensure_idea_preview_table(con: sqlite3.Connection) -> None:
    """Cache only verified real catalog images; discard the old AI-image schema."""
    existing = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='idea_previews'"
    ).fetchone()
    if existing:
        columns = {
            row[1] for row in con.execute("PRAGMA table_info(idea_previews)")
        }
        required = {"id", "set_num", "source_url", "image", "content_type"}
        if not required.issubset(columns):
            con.execute("DROP TABLE idea_previews")
    con.execute("""
        CREATE TABLE IF NOT EXISTS idea_previews (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            set_num TEXT NOT NULL,
            source_url TEXT NOT NULL,
            image BLOB NOT NULL,
            content_type TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def ensure_derived() -> None:
    """Derived tables/indexes the ingest may not have built yet."""
    with closing(db()) as con:
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "set_parts" not in have:
            log.warning("set_parts missing - run `python3 ingest.py` first")
            return
        con.executescript("""
            CREATE INDEX IF NOT EXISTS idx_ip_pc ON inventory_parts(part_num, color_id);
            CREATE TABLE IF NOT EXISTS my_parts (
                part_num TEXT NOT NULL, color_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                added_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (part_num, color_id));
        """)
        ensure_idea_preview_table(con)
        if "set_parts_any" not in have:
            log.info("building set_parts_any (color-agnostic) ...")
            con.executescript("""
                CREATE TABLE set_parts_any AS
                SELECT set_num, part_num, SUM(quantity) AS quantity
                FROM set_parts GROUP BY set_num, part_num;
                CREATE INDEX idx_spa_set ON set_parts_any(set_num);
                CREATE INDEX idx_spa_part ON set_parts_any(part_num);
            """)
        con.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_derived()
    yield


app = FastAPI(title="mylego-vision", lifespan=lifespan)


# ---------------------------------------------------------------- helpers

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def dominant_color(img: Image.Image) -> tuple[int, int, int] | None:
    """Dominant color of the central region, background pixels filtered out."""
    im = img.convert("RGB").resize((96, 96))
    w, h = im.size
    crop = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    pixels = getattr(crop, "get_flattened_data", crop.getdata)
    px = list(pixels())
    keep = []
    for r, g, b in px:
        mx, mn = max(r, g, b), min(r, g, b)
        if mx > 235 and mn > 215:          # near-white background
            continue
        keep.append((r, g, b))
    if len(keep) < 20:
        keep = px
    if not keep:
        return None
    n = len(keep)
    return (sum(p[0] for p in keep) // n,
            sum(p[1] for p in keep) // n,
            sum(p[2] for p in keep) // n)


_palette: list[sqlite3.Row] = []


def nearest_lego_colors(rgb: tuple[int, int, int], k: int = 3) -> list[dict]:
    """Nearest LEGO colors, restricted to colors that actually appear in real
    sets. A frequency prior resolves camera/lighting ambiguity in favor of
    common solid colors; special finishes are excluded from photo matching."""
    global _palette
    if not _palette:
        with closing(db()) as con:
            _palette = con.execute("""
                SELECT c.id, c.name, c.rgb, COUNT(*) AS n
                FROM colors c JOIN set_parts sp ON sp.color_id = c.id
                WHERE c.id >= 0 AND c.is_trans IN ('f','False')
                GROUP BY c.id HAVING n >= 500""").fetchall()
    mx, mn = max(rgb), min(rgb)
    saturation = (mx - mn) / mx if mx else 0
    forced = None
    if saturation < 0.12 and mx > 170:
        forced = "White"
    elif saturation < 0.15 and mx < 90:
        forced = "Black"
    scored = []
    for r in _palette:
        if any(marker in r["name"].lower() for marker in SPECIAL_FINISH_MARKERS):
            continue
        cr, cg, cb = _hex_to_rgb(r["rgb"])
        d = (cr - rgb[0]) ** 2 + (cg - rgb[1]) ** 2 + (cb - rgb[2]) ** 2
        score = d / max(r["n"], 1) ** COLOR_POPULARITY_POWER
        if r["name"] == forced:
            score = -1
        scored.append((score, {
            "color_id": r["id"], "name": r["name"], "rgb": r["rgb"],
        }))
    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored[:k]]


async def ollama_chat(model: str, prompt: str, image_b64: str | None = None,
                      json_mode: bool = False, timeout: float = 180.0) -> str:
    msg: dict = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    body: dict = {"model": model, "messages": [msg], "stream": False, "think": False,
                  "options": {"temperature": 0.4}}
    if json_mode:
        body["format"] = "json"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/chat", json=body)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
    return THINK_RE.sub("", content).strip()


def parse_json_loose(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"[\[{]", text)):
        try:
            value, _ = decoder.raw_decode(text, start)
            return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no valid JSON object or array found", text, 0)


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _edge_background(img: Image.Image) -> tuple[tuple[int, int, int], int]:
    """Estimate a plain tabletop/background color and its natural variation."""
    width, height = img.size
    step = max(1, min(width, height) // 160)
    pixels = img.load()
    samples = []
    for x in range(0, width, step):
        samples.extend((pixels[x, 0], pixels[x, height - 1]))
    for y in range(step, height - 1, step):
        samples.extend((pixels[0, y], pixels[width - 1, y]))
    background = tuple(_median([p[channel] for p in samples]) for channel in range(3))
    deviations = sorted(max(abs(p[i] - background[i]) for i in range(3)) for p in samples)
    edge_noise = deviations[min(len(deviations) - 1, int(len(deviations) * 0.85))]
    return background, edge_noise


def _connected_boxes(mask: Image.Image, min_area: int) -> list[tuple[int, int, int, int, int]]:
    """Return 8-connected foreground boxes as (area, left, top, right, bottom)."""
    width, height = mask.size
    foreground = bytearray(mask.tobytes())
    boxes = []
    for start in range(width * height):
        if not foreground[start]:
            continue
        foreground[start] = 0
        stack = [start]
        area = 0
        left = right = start % width
        top = bottom = start // width
        while stack:
            pos = stack.pop()
            x, y = pos % width, pos // width
            area += 1
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                row = ny * width
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbour = row + nx
                    if foreground[neighbour]:
                        foreground[neighbour] = 0
                        stack.append(neighbour)
        if area >= min_area:
            boxes.append((area, left, top, right + 1, bottom + 1))
    return boxes


def extract_scattered_parts(raw: bytes, max_parts: int = 12) -> list[dict]:
    """Extract non-touching LEGO pieces from a photo taken on a plain surface.

    This deliberately uses Pillow rather than a bundled ML detector. It keeps the
    feature dependency- and model-license-free, while making the limitation clear:
    touching or overlapping bricks can be returned as one region.
    """
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid image") from exc
    if source.width < 80 or source.height < 80:
        raise ValueError("image is too small")

    working = source.copy()
    working.thumbnail((PILE_WORKING_SIDE, PILE_WORKING_SIDE), Image.Resampling.LANCZOS)
    background, edge_noise = _edge_background(working)
    flat_background = Image.new("RGB", working.size, background)
    red, green, blue = ImageChops.difference(working, flat_background).split()
    difference = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    threshold = min(96, max(28, edge_noise + 18))
    mask = difference.point(lambda value: 255 if value >= threshold else 0)
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))

    image_area = working.width * working.height
    components = _connected_boxes(mask, max(80, int(image_area * 0.00055)))
    useful = []
    for component in components:
        area, left, top, right, bottom = component
        box_width, box_height = right - left, bottom - top
        box_area = box_width * box_height
        if min(box_width, box_height) < 12 or box_area > image_area * 0.88:
            continue
        if area / box_area < 0.06:
            continue
        useful.append(component)

    # Prefer the largest real regions when noise produces more than the limit,
    # then present them in natural reading order.
    useful.sort(reverse=True)
    useful = useful[:max_parts]
    useful.sort(key=lambda item: (item[2], item[1]))

    scale_x = source.width / working.width
    scale_y = source.height / working.height
    regions = []
    for _, left, top, right, bottom in useful:
        margin = max(8, int(max(right - left, bottom - top) * 0.09))
        left, top = max(0, left - margin), max(0, top - margin)
        right, bottom = min(working.width, right + margin), min(working.height, bottom + margin)
        original_box = (
            max(0, int(left * scale_x)),
            max(0, int(top * scale_y)),
            min(source.width, int(right * scale_x + 0.999)),
            min(source.height, int(bottom * scale_y + 0.999)),
        )
        regions.append({"bbox": original_box, "image": source.crop(original_box)})
    return regions


def _jpeg_bytes(img: Image.Image, max_side: int = 900, quality: int = 88) -> bytes:
    prepared = img.copy()
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    prepared.save(encoded, format="JPEG", quality=quality, optimize=True)
    return encoded.getvalue()


async def _brickognize_items(
    client: httpx.AsyncClient, raw: bytes, filename: str, content_type: str
) -> list[dict]:
    response = await client.post(
        BRICKOGNIZE_URL,
        files={"query_image": (filename, raw, content_type)},
    )
    response.raise_for_status()
    return response.json().get("items", [])


def _catalog_candidates(con: sqlite3.Connection, items: list[dict], limit: int = 6) -> list[dict]:
    candidates = []
    for item in items[:limit]:
        part_num = str(item.get("id", ""))
        known = con.execute(
            "SELECT part_num FROM parts WHERE part_num = ?", (part_num,)
        ).fetchone()
        candidates.append({
            "part_num": part_num,
            "name": item.get("name"),
            "type": item.get("type"),
            "score": round(float(item.get("score") or 0), 3),
            "img_url": item.get("img_url"),
            "in_db": bool(known),
        })
    return candidates


# ---------------------------------------------------------------- status

@app.get("/api/status")
async def status():
    counts = {}
    with closing(db()) as con:
        for t in ("parts", "colors", "sets", "set_parts", "my_parts"):
            try:
                counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[t] = None
    ollama_ok, models = False, []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except Exception:
        pass
    return {"db": counts, "ollama": {"host": OLLAMA_HOST, "ok": ollama_ok,
            "vision_model": OLLAMA_VISION_MODEL, "text_model": OLLAMA_TEXT_MODEL,
            "models": models}}


# ---------------------------------------------------------------- scan (MVP 1)

@app.post("/api/scan")
async def scan(
    image: UploadFile = File(...),
    engine: str = Query("fast", pattern="^(fast|deep)$"),
):
    """Identify one LEGO part on a photo.
    fast = Brickognize + local dominant-color -> LEGO palette
    deep = fast + Ollama vision model second opinion"""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image")

    result: dict = {"candidates": [], "colors": [], "ollama": None}

    # local color detection
    try:
        img = Image.open(io.BytesIO(raw))
        rgb = dominant_color(img)
        if rgb:
            result["detected_rgb"] = "%02X%02X%02X" % rgb
            result["colors"] = nearest_lego_colors(rgb)
    except Exception as e:
        log.warning("color detect failed: %s", e)

    # Brickognize - part identification
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            items = await _brickognize_items(
                client,
                raw,
                image.filename or "part.jpg",
                image.content_type or "image/jpeg",
            )
        with closing(db()) as con:
            result["candidates"] = _catalog_candidates(con, items)
    except Exception as e:
        log.warning("brickognize failed: %s", e)
        result["brickognize_error"] = str(e)

    # Ollama vision second opinion
    if engine == "deep":
        try:
            b64 = base64.b64encode(raw).decode()
            txt = await ollama_chat(
                OLLAMA_VISION_MODEL,
                "You are a LEGO expert. Look at this photo of a single LEGO part. "
                "Reply with strict JSON: {\"description\": short part description, "
                "\"guess_part\": likely LDraw/Rebrickable part number or null, "
                "\"color\": common LEGO color name, \"stud_size\": e.g. '2x4' or null}",
                image_b64=b64, json_mode=True)
            result["ollama"] = parse_json_loose(txt)
        except Exception as e:
            log.warning("ollama vision failed: %s", e)
            result["ollama"] = {"error": str(e)}

    return result


@app.post("/api/scan/pile")
async def scan_pile(
    image: UploadFile = File(...),
    max_parts: int = Query(12, ge=1, le=PILE_MAX_PARTS),
):
    """Identify separated, non-touching LEGO parts on a plain background."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image")
    try:
        regions = extract_scattered_parts(raw, max_parts=max_parts)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not regions:
        return {
            "parts": [],
            "warning": "No separate pieces found. Use a plain contrasting surface and leave gaps between pieces.",
        }

    semaphore = asyncio.Semaphore(3)

    async with httpx.AsyncClient(timeout=30) as client:
        async def recognize(index: int, region: dict) -> dict:
            crop = region["image"]
            crop_bytes = _jpeg_bytes(crop)
            preview_bytes = _jpeg_bytes(crop, max_side=320, quality=78)
            result = {
                "index": index,
                "bbox": list(region["bbox"]),
                "preview": "data:image/jpeg;base64," + base64.b64encode(preview_bytes).decode(),
                "colors": [],
                "items": [],
            }
            rgb = dominant_color(crop)
            if rgb:
                result["detected_rgb"] = "%02X%02X%02X" % rgb
                result["colors"] = nearest_lego_colors(rgb)
            try:
                async with semaphore:
                    result["items"] = await _brickognize_items(
                        client, crop_bytes, f"pile-part-{index}.jpg", "image/jpeg"
                    )
            except Exception as exc:
                log.warning("brickognize pile crop %s failed: %s", index, exc)
                result["error"] = str(exc)
            return result

        recognized = await asyncio.gather(*(
            recognize(index, region) for index, region in enumerate(regions, start=1)
        ))

    with closing(db()) as con:
        for part in recognized:
            part["candidates"] = _catalog_candidates(con, part.pop("items"), limit=4)

    return {
        "parts": recognized,
        "warning": (
            "Touching or overlapping pieces may be merged. Check every suggestion before adding it."
        ),
    }


# ---------------------------------------------------------------- inventory

class InvItem(BaseModel):
    part_num: str
    color_id: int
    quantity: int = Field(default=1, ge=1)


@app.get("/api/inventory")
def get_inventory():
    with closing(db()) as con:
        rows = con.execute("""
            SELECT mp.part_num, mp.color_id, mp.quantity,
                   p.name AS part_name, c.name AS color_name, c.rgb,
                   (SELECT img_url FROM inventory_parts ip
                    WHERE ip.part_num = mp.part_num AND ip.color_id = mp.color_id
                      AND ip.img_url != '' LIMIT 1) AS img_url
            FROM my_parts mp
            LEFT JOIN parts p ON p.part_num = mp.part_num
            LEFT JOIN colors c ON c.id = mp.color_id
            ORDER BY mp.added_at DESC""").fetchall()
        total = con.execute("SELECT COALESCE(SUM(quantity),0) FROM my_parts").fetchone()[0]
    return {"total": total, "items": [dict(r) for r in rows]}


@app.post("/api/inventory")
def add_inventory(item: InvItem):
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM parts WHERE part_num=?", (item.part_num,)).fetchone():
            raise HTTPException(404, f"unknown part {item.part_num}")
        if not con.execute("SELECT 1 FROM colors WHERE id=?", (item.color_id,)).fetchone():
            raise HTTPException(404, f"unknown color {item.color_id}")
        con.execute("""
            INSERT INTO my_parts (part_num, color_id, quantity) VALUES (?,?,?)
            ON CONFLICT(part_num, color_id)
            DO UPDATE SET quantity = quantity + excluded.quantity""",
            (item.part_num, item.color_id, item.quantity))
        con.commit()
    return {"ok": True}


@app.post("/api/inventory/bulk")
def add_inventory_bulk(items: list[InvItem]):
    """Atomically add user-confirmed results from a scattered-parts scan."""
    if not items:
        raise HTTPException(400, "no parts selected")
    if len(items) > PILE_MAX_PARTS:
        raise HTTPException(400, f"at most {PILE_MAX_PARTS} parts can be added at once")

    combined: dict[tuple[str, int], int] = {}
    for item in items:
        key = (item.part_num, item.color_id)
        combined[key] = combined.get(key, 0) + item.quantity

    with closing(db()) as con:
        for part_num, color_id in combined:
            if not con.execute("SELECT 1 FROM parts WHERE part_num=?", (part_num,)).fetchone():
                raise HTTPException(404, f"unknown part {part_num}")
            if not con.execute("SELECT 1 FROM colors WHERE id=?", (color_id,)).fetchone():
                raise HTTPException(404, f"unknown color {color_id}")
        con.executemany(
            """INSERT INTO my_parts (part_num, color_id, quantity) VALUES (?,?,?)
               ON CONFLICT(part_num, color_id)
               DO UPDATE SET quantity = quantity + excluded.quantity""",
            [(part_num, color_id, quantity)
             for (part_num, color_id), quantity in combined.items()],
        )
        con.commit()
    return {
        "ok": True,
        "distinct_items": len(combined),
        "added_quantity": sum(combined.values()),
    }


@app.post("/api/inventory/import-set/{set_num}")
def import_set_inventory(set_num: str):
    """Add the complete latest inventory of an owned set to my_parts."""
    with closing(db()) as con:
        lego_set = con.execute(
            "SELECT set_num, name FROM sets WHERE set_num=?", (set_num,)
        ).fetchone()
        if not lego_set:
            raise HTTPException(404, f"unknown set {set_num}")

        totals = con.execute(
            """SELECT COUNT(*) AS distinct_items,
                      COALESCE(SUM(quantity), 0) AS total_quantity
               FROM set_parts WHERE set_num=?""",
            (set_num,),
        ).fetchone()
        if totals["total_quantity"] == 0:
            raise HTTPException(404, f"set {set_num} has no inventory")

        con.execute(
            """INSERT INTO my_parts (part_num, color_id, quantity)
               SELECT part_num, color_id, quantity
               FROM set_parts
               WHERE set_num=?
               ON CONFLICT(part_num, color_id)
               DO UPDATE SET quantity = quantity + excluded.quantity""",
            (set_num,),
        )
        con.commit()
    return {
        "ok": True,
        "set_num": lego_set["set_num"],
        "name": lego_set["name"],
        "added_quantity": totals["total_quantity"],
        "distinct_items": totals["distinct_items"],
    }


@app.delete("/api/inventory/{part_num}/{color_id}")
def del_inventory(part_num: str, color_id: int):
    with closing(db()) as con:
        con.execute("DELETE FROM my_parts WHERE part_num=? AND color_id=?", (part_num, color_id))
        con.commit()
    return {"ok": True}


@app.post("/api/inventory/clear")
def clear_inventory():
    with closing(db()) as con:
        con.execute("DELETE FROM my_parts")
        con.commit()
    return {"ok": True}


@app.get("/api/parts/search")
def search_parts(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100),
):
    with closing(db()) as con:
        rows = con.execute("""
            SELECT part_num, name FROM parts
            WHERE part_num LIKE ? OR name LIKE ?
            ORDER BY LENGTH(part_num) LIMIT ?""",
            (f"{q}%", f"%{q}%", limit)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/colors")
def colors():
    with closing(db()) as con:
        rows = con.execute(
            "SELECT id, name, rgb FROM colors WHERE id >= 0 ORDER BY name").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/sets/search")
def search_sets(
    q: str = Query(..., min_length=2),
    limit: int = Query(8, ge=1, le=20),
):
    with closing(db()) as con:
        rows = con.execute(
            """SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url,
                      st.total_qty
               FROM sets s
               JOIN set_totals st ON st.set_num = s.set_num
               WHERE s.set_num LIKE ? OR s.name LIKE ?
               ORDER BY CASE
                   WHEN LOWER(s.set_num) = LOWER(?) THEN 0
                   WHEN s.set_num LIKE ? THEN 1
                   ELSE 2
               END, st.total_qty DESC
               LIMIT ?""",
            (f"{q}%", f"%{q}%", q, f"{q}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- buildable (MVP 2)

@app.get("/api/buildable")
def buildable(mode: str = Query("strict", pattern="^(strict|loose)$"),
              limit: int = Query(30, ge=1, le=100)):
    """Rank real LEGO sets by how buildable they are from my_parts.
    strict = part+color must match, loose = shape only, any color."""
    with closing(db()) as con:
        user_total = con.execute("SELECT COALESCE(SUM(quantity),0) FROM my_parts").fetchone()[0]
        if user_total == 0:
            return {"user_total": 0, "sets": []}
        cap = max(user_total * 5, 50)
        if mode == "strict":
            sql = """
                WITH cov AS (
                    SELECT sp.set_num,
                           SUM(MIN(sp.quantity, COALESCE(mp.quantity, 0))) AS have,
                           SUM(sp.quantity) AS need
                    FROM set_parts sp
                    JOIN set_totals st ON st.set_num = sp.set_num
                    LEFT JOIN my_parts mp
                         ON mp.part_num = sp.part_num AND mp.color_id = sp.color_id
                    WHERE st.total_qty BETWEEN :minp AND :cap
                    GROUP BY sp.set_num
                    HAVING have > 0
                )"""
        else:
            sql = """
                WITH mine AS (SELECT part_num, SUM(quantity) AS q
                              FROM my_parts GROUP BY part_num),
                cov AS (
                    SELECT sp.set_num,
                           SUM(MIN(sp.quantity, COALESCE(mine.q, 0))) AS have,
                           SUM(sp.quantity) AS need
                    FROM set_parts_any sp
                    JOIN set_totals st ON st.set_num = sp.set_num
                    LEFT JOIN mine ON mine.part_num = sp.part_num
                    WHERE st.total_qty BETWEEN :minp AND :cap
                    GROUP BY sp.set_num
                    HAVING have > 0
                )"""
        sql += """
            SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url,
                   t.name AS theme, cov.have, cov.need,
                   ROUND(100.0 * cov.have / cov.need, 1) AS pct
            FROM cov
            JOIN sets s ON s.set_num = cov.set_num
            LEFT JOIN themes t ON t.id = s.theme_id
            ORDER BY 1.0 * cov.have / cov.need DESC, cov.need DESC
            LIMIT :limit"""
        rows = con.execute(sql, {"minp": MIN_SET_PARTS, "cap": cap,
                                 "limit": limit}).fetchall()
    return {"user_total": user_total, "mode": mode, "sets": [dict(r) for r in rows]}


@app.get("/api/buildable/{set_num}/missing")
def missing_parts(set_num: str, mode: str = Query("strict", pattern="^(strict|loose)$")):
    with closing(db()) as con:
        if mode == "strict":
            rows = con.execute("""
                SELECT sp.part_num, sp.color_id, sp.quantity AS need,
                       COALESCE(mp.quantity, 0) AS have,
                       p.name AS part_name, c.name AS color_name, c.rgb,
                       (SELECT img_url FROM inventory_parts ip
                        WHERE ip.part_num = sp.part_num AND ip.color_id = sp.color_id
                          AND ip.img_url != '' LIMIT 1) AS img_url
                FROM set_parts sp
                LEFT JOIN my_parts mp
                     ON mp.part_num = sp.part_num AND mp.color_id = sp.color_id
                LEFT JOIN parts p ON p.part_num = sp.part_num
                LEFT JOIN colors c ON c.id = sp.color_id
                WHERE sp.set_num = ?
                ORDER BY (sp.quantity - COALESCE(mp.quantity,0)) DESC""",
                (set_num,)).fetchall()
        else:
            rows = con.execute("""
                WITH mine AS (SELECT part_num, SUM(quantity) AS q
                              FROM my_parts GROUP BY part_num)
                SELECT sp.part_num, NULL AS color_id, sp.quantity AS need,
                       COALESCE(mine.q, 0) AS have,
                       p.name AS part_name, NULL AS color_name, NULL AS rgb,
                       (SELECT img_url FROM inventory_parts ip
                        WHERE ip.part_num = sp.part_num AND ip.img_url != ''
                        LIMIT 1) AS img_url
                FROM set_parts_any sp
                LEFT JOIN mine ON mine.part_num = sp.part_num
                LEFT JOIN parts p ON p.part_num = sp.part_num
                WHERE sp.set_num = ?
                ORDER BY (sp.quantity - COALESCE(mine.q,0)) DESC""",
                (set_num,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- AI advisor

class IdeaPreviewRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    reference_set_num: str | None = Field(default=None, max_length=30)


def _normalize_set_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _find_real_idea_set(
    con: sqlite3.Connection, idea: IdeaPreviewRequest
) -> sqlite3.Row | None:
    title = idea.title.strip()
    normalized_title = _normalize_set_title(title)
    if not normalized_title:
        return None

    if idea.reference_set_num:
        set_num = idea.reference_set_num.strip()
        referenced = con.execute(
            """SELECT set_num, name, img_url FROM sets
               WHERE set_num=? AND img_url != ''""",
            (set_num,),
        ).fetchone()
        if referenced and (
            set_num.lower() in title.lower()
            or _normalize_set_title(referenced["name"]) == normalized_title
        ):
            return referenced

    return con.execute(
        """SELECT set_num, name, img_url FROM sets
           WHERE LOWER(TRIM(name)) = LOWER(?) AND img_url != ''
           ORDER BY year DESC, num_parts DESC
           LIMIT 1""",
        (title,),
    ).fetchone()


def _placeholder_preview(reason: str = "no_real_match") -> dict:
    return {
        "found": False,
        "preview_url": IDEA_PLACEHOLDER_URL,
        "download_url": None,
        "reason": reason,
    }


def _idea_preview_payload(row: sqlite3.Row, cached: bool) -> dict:
    preview_id = row["id"]
    return {
        "found": True,
        "id": preview_id,
        "set_num": row["set_num"],
        "set_name": row["title"],
        "width": row["width"],
        "height": row["height"],
        "preview_url": f"/api/ideas/preview/{preview_id}",
        "download_url": f"/api/ideas/preview/{preview_id}/download",
        "cached": cached,
    }


async def _download_real_idea_preview(
    source_url: str,
) -> tuple[bytes, str, int, int]:
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or parsed.hostname not in REAL_IMAGE_HOSTS:
        raise ValueError("unsupported real-image source")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(
            source_url,
            headers={
                "Accept": "image/jpeg,image/png,image/webp",
                "User-Agent": "mylego-vision/1.0",
            },
        )
        response.raise_for_status()
        raw = response.content

    if not raw or len(raw) > IDEA_PREVIEW_MAX_BYTES:
        raise ValueError("preview image is empty or too large")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except (OSError, ValueError) as exc:
        raise ValueError("preview provider returned an invalid image") from exc

    content_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    content_type = content_types.get(image_format)
    if not content_type or width < 256 or height < 256:
        raise ValueError("preview provider returned an unsupported image")
    return raw, content_type, width, height


@app.post("/api/ideas/preview")
async def create_idea_preview(idea: IdeaPreviewRequest):
    with closing(db()) as con:
        ensure_idea_preview_table(con)
        con.commit()
        lego_set = _find_real_idea_set(con, idea)
        if not lego_set:
            return _placeholder_preview()
        preview_id = hashlib.sha256(
            lego_set["img_url"].encode("utf-8")
        ).hexdigest()[:24]
        cached = con.execute(
            """SELECT id, title, set_num, width, height
               FROM idea_previews WHERE id=?""",
            (preview_id,),
        ).fetchone()
    if cached:
        return _idea_preview_payload(cached, cached=True)

    try:
        raw, content_type, width, height = await _download_real_idea_preview(
            lego_set["img_url"]
        )
    except (httpx.HTTPError, OSError, ValueError) as exc:
        log.warning("real idea preview download failed: %s", exc)
        return _placeholder_preview("real_image_unavailable")

    with closing(db()) as con:
        ensure_idea_preview_table(con)
        con.execute(
            """INSERT OR REPLACE INTO idea_previews
                   (id, title, set_num, source_url, image,
                    content_type, width, height)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                preview_id,
                lego_set["name"],
                lego_set["set_num"],
                lego_set["img_url"],
                sqlite3.Binary(raw),
                content_type,
                width,
                height,
            ),
        )
        con.execute(
            """DELETE FROM idea_previews
               WHERE id NOT IN (
                   SELECT id FROM idea_previews
                   ORDER BY created_at DESC, rowid DESC
                   LIMIT ?
               )""",
            (IDEA_PREVIEW_CACHE_LIMIT,),
        )
        con.commit()
        stored = con.execute(
            """SELECT id, title, set_num, width, height
               FROM idea_previews WHERE id=?""",
            (preview_id,),
        ).fetchone()
    return _idea_preview_payload(stored, cached=False)


def _stored_idea_preview(preview_id: str) -> sqlite3.Row:
    if not re.fullmatch(r"[0-9a-f]{24}", preview_id):
        raise HTTPException(404, "preview not found")
    with closing(db()) as con:
        ensure_idea_preview_table(con)
        con.commit()
        row = con.execute(
            """SELECT image, content_type, set_num
               FROM idea_previews WHERE id=?""",
            (preview_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "preview not found")
    return row


@app.get("/api/ideas/preview/{preview_id}")
def get_idea_preview(preview_id: str):
    row = _stored_idea_preview(preview_id)
    return Response(
        content=bytes(row["image"]),
        media_type=row["content_type"],
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/ideas/preview/{preview_id}/download")
def download_idea_preview(preview_id: str):
    row = _stored_idea_preview(preview_id)
    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }[row["content_type"]]
    safe_set_num = re.sub(r"[^A-Za-z0-9_-]", "_", row["set_num"])
    return Response(
        content=bytes(row["image"]),
        media_type=row["content_type"],
        headers={
            "Content-Disposition": (
                f'attachment; filename="lego-set-{safe_set_num}.{extension}"'
            ),
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@app.post("/api/advise")
async def advise():
    """Ask the local LLM for creative MOC ideas from the current inventory."""
    with closing(db()) as con:
        rows = con.execute("""
            SELECT mp.quantity, p.name AS part_name, c.name AS color_name
            FROM my_parts mp
            LEFT JOIN parts p ON p.part_num = mp.part_num
            LEFT JOIN colors c ON c.id = mp.color_id
            ORDER BY mp.quantity DESC LIMIT 60""").fetchall()
    if not rows:
        raise HTTPException(400, "inventory is empty - scan some parts first")

    inv_lines = "\n".join(
        f"- {r['quantity']}x {r['part_name'] or 'unknown part'} ({r['color_name'] or '?'})"
        for r in rows)
    prompt = (
        "You are a master LEGO builder. Here is my LEGO parts inventory:\n"
        f"{inv_lines}\n\n"
        "Suggest 4 creative small models (MOCs) I could realistically build using ONLY "
        "these parts (or a subset). Consider part shapes and colors. "
        "Reply with strict JSON: {\"ideas\": [{\"title\": str, \"emoji\": one emoji, "
        "\"description\": 2 sentences how to build it, \"parts_used\": [str, ...], "
        "\"difficulty\": \"easy\"|\"medium\"|\"hard\"}]}")
    try:
        txt = await ollama_chat(OLLAMA_TEXT_MODEL, prompt, json_mode=True, timeout=300)
        data = parse_json_loose(txt)
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
        raise HTTPException(502, f"ollama advise failed: {e}")
    return data


# ---------------------------------------------------------------- static

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
