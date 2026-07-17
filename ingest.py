# mylego-vision - Rebrickable dataset ingest
# downloads the official Rebrickable CSV dumps and loads them into SQLite,
# then precomputes set_parts (parts per set, latest inventory, no spares)
# so the buildability engine can run fast queries.
# copyright (c) 2026 cocomelonc
import argparse
import csv
import gzip
import io
import logging
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("ingest")

_configured_db = Path(os.getenv("LEGO_DB", "lego.db"))
DB_PATH = str(
    _configured_db if _configured_db.is_absolute() else BASE_DIR / _configured_db
)
CDN = "https://cdn.rebrickable.com/media/downloads"

# table -> (columns, primary key sql)
TABLES = {
    "colors": ("id,name,rgb,is_trans", "id INTEGER PRIMARY KEY, name TEXT, rgb TEXT, is_trans TEXT"),
    "part_categories": ("id,name", "id INTEGER PRIMARY KEY, name TEXT"),
    "parts": ("part_num,name,part_cat_id,part_material",
              "part_num TEXT PRIMARY KEY, name TEXT, part_cat_id INTEGER, part_material TEXT"),
    "themes": ("id,name,parent_id", "id INTEGER PRIMARY KEY, name TEXT, parent_id INTEGER"),
    "sets": ("set_num,name,year,theme_id,num_parts,img_url",
             "set_num TEXT PRIMARY KEY, name TEXT, year INTEGER, theme_id INTEGER, num_parts INTEGER, img_url TEXT"),
    "inventories": ("id,version,set_num", "id INTEGER PRIMARY KEY, version INTEGER, set_num TEXT"),
    "inventory_parts": ("inventory_id,part_num,color_id,quantity,is_spare,img_url",
                        "inventory_id INTEGER, part_num TEXT, color_id INTEGER, quantity INTEGER, is_spare TEXT, img_url TEXT"),
}


def download(name: str) -> bytes:
    url = f"{CDN}/{name}.csv.gz"
    log.info("downloading %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "mylego-vision/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    log.info("  %s: %.1f MB gz", name, len(data) / 1e6)
    return data


def load_table(con: sqlite3.Connection, name: str, gz: bytes) -> None:
    cols, schema = TABLES[name]
    col_list = cols.split(",")
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} ({schema})")
    text = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(gz)), encoding="utf-8")
    reader = csv.DictReader(text)
    placeholders = ",".join("?" * len(col_list))
    rows = ([row.get(c) for c in col_list] for row in reader)
    cur = con.executemany(f"INSERT INTO {name} ({cols}) VALUES ({placeholders})", rows)
    con.commit()
    n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    log.info("  %s: %d rows", name, n)


def precompute(con: sqlite3.Connection) -> None:
    """Build set part requirements from the latest inventory, excluding spares."""
    log.info("precomputing set_parts ...")
    try:
        con.executescript("""
            BEGIN IMMEDIATE;
            CREATE INDEX IF NOT EXISTS idx_inv_set ON inventories(set_num, version);
            CREATE INDEX IF NOT EXISTS idx_ip_inv ON inventory_parts(inventory_id);

            DROP TABLE IF EXISTS set_parts;
            CREATE TABLE set_parts AS
            SELECT i.set_num,
                   ip.part_num,
                   ip.color_id,
                   SUM(ip.quantity) AS quantity
            FROM inventory_parts ip
            JOIN inventories i ON i.id = ip.inventory_id
            JOIN (SELECT set_num, MAX(version) AS v FROM inventories GROUP BY set_num) lv
                 ON lv.set_num = i.set_num AND lv.v = i.version
            WHERE ip.is_spare IN ('f', 'False')
            GROUP BY i.set_num, ip.part_num, ip.color_id;

            CREATE INDEX idx_sp_set  ON set_parts(set_num);
            CREATE INDEX idx_sp_part ON set_parts(part_num, color_id);

            DROP TABLE IF EXISTS set_totals;
            CREATE TABLE set_totals AS
            SELECT set_num, SUM(quantity) AS total_qty, COUNT(*) AS distinct_parts
            FROM set_parts GROUP BY set_num;
            CREATE UNIQUE INDEX idx_st_set ON set_totals(set_num);

            DROP TABLE IF EXISTS set_parts_any;
            CREATE TABLE set_parts_any AS
            SELECT set_num, part_num, SUM(quantity) AS quantity
            FROM set_parts GROUP BY set_num, part_num;
            CREATE INDEX idx_spa_set ON set_parts_any(set_num);
            CREATE INDEX idx_spa_part ON set_parts_any(part_num);

            -- user inventory lives here
            CREATE TABLE IF NOT EXISTS my_parts (
                part_num TEXT NOT NULL,
                color_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                added_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (part_num, color_id)
            );
            COMMIT;
        """)
    except Exception:
        con.rollback()
        raise
    n = con.execute("SELECT COUNT(*) FROM set_parts").fetchone()[0]
    log.info("  set_parts: %d rows", n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the local Rebrickable database")
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="rebuild derived set tables from existing raw tables without downloading",
    )
    args = parser.parse_args(argv)

    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        if not args.precompute_only:
            con.execute("PRAGMA synchronous=OFF")
            for name in TABLES:
                load_table(con, name, download(name))
        precompute(con)
        if not args.precompute_only:
            con.execute("PRAGMA synchronous=NORMAL")
    finally:
        con.close()
    log.info("done -> %s", DB_PATH)


if __name__ == "__main__":
    sys.exit(main())
