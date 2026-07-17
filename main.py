# mylego-vision - what can I build from my LEGO parts?
# scan parts with CV (Brickognize + Ollama vision), keep an inventory,
# rank every real LEGO set by buildability, ask a local LLM for MOC ideas.
# copyright (c) 2026 cocomelonc
# author: cocomelonc
import base64
import io
import json
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
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
            r = await client.post(
                BRICKOGNIZE_URL,
                files={"query_image": (image.filename or "part.jpg", raw,
                                       image.content_type or "image/jpeg")})
            r.raise_for_status()
            items = r.json().get("items", [])
        with closing(db()) as con:
            for it in items[:6]:
                pid = it.get("id", "")
                known = con.execute(
                    "SELECT part_num FROM parts WHERE part_num = ?", (pid,)).fetchone()
                result["candidates"].append({
                    "part_num": pid,
                    "name": it.get("name"),
                    "type": it.get("type"),
                    "score": round(it.get("score", 0), 3),
                    "img_url": it.get("img_url"),
                    "in_db": bool(known),
                })
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
