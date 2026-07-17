import os
import io
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

import ingest
import main


class ColorDetectionRegressionTests(unittest.TestCase):
    def test_sample_photos_rank_their_known_solid_color_first(self):
        palette = [
            {"id": 0, "name": "Black", "rgb": "05131D", "n": 226411},
            {"id": 1, "name": "Blue", "rgb": "0055BF", "n": 48865},
            {"id": 2, "name": "Green", "rgb": "237841", "n": 26475},
            {"id": 4, "name": "Red", "rgb": "C91A09", "n": 94059},
            {"id": 14, "name": "Yellow", "rgb": "F2CD37", "n": 69629},
            {"id": 15, "name": "White", "rgb": "FFFFFF", "n": 148761},
            {"id": 82, "name": "Metallic Gold", "rgb": "DBAC34", "n": 745},
            {"id": 85, "name": "Dark Purple", "rgb": "3F3691", "n": 5809},
            {"id": 191, "name": "Bright Light Orange", "rgb": "F8BB3D", "n": 10045},
            {"id": 323, "name": "Light Aqua", "rgb": "ADC3C0", "n": 3101},
            {"id": 1103, "name": "Pearl Titanium", "rgb": "3E3C39", "n": 3105},
            {"id": 1136, "name": "Reddish Orange", "rgb": "CA4C0B", "n": 812},
        ]
        expected = {
            "brick-1x4-yellow.jpg": "Yellow",
            "brick-2x2-blue.jpg": "Blue",
            "brick-2x4-red.jpg": "Red",
            "brick-2x4-white.jpg": "White",
            "headlight-black.jpg": "Black",
            "plate-1x2-black.jpg": "Black",
            "plate-2x4-red.jpg": "Red",
            "round-brick-yellow.jpg": "Yellow",
            "round-plate-green.jpg": "Green",
            "slope-2x2-blue.jpg": "Blue",
        }
        image_dir = Path(main.__file__).resolve().parent / "test_images"
        old_palette = main._palette
        main._palette = palette
        try:
            for filename, color_name in expected.items():
                with self.subTest(filename=filename):
                    with Image.open(image_dir / filename) as image:
                        rgb = main.dominant_color(image)
                    actual = main.nearest_lego_colors(rgb, k=1)[0]["name"]
                    self.assertEqual(actual, color_name)
        finally:
            main._palette = old_palette


class PrecomputeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(
            """
            CREATE TABLE inventories (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                set_num TEXT
            );
            CREATE TABLE inventory_parts (
                inventory_id INTEGER,
                part_num TEXT,
                color_id INTEGER,
                quantity INTEGER,
                is_spare TEXT,
                img_url TEXT
            );
            """
        )

    def tearDown(self):
        self.con.close()

    def test_precompute_uses_latest_inventory_version(self):
        self.con.executescript(
            """
            INSERT INTO inventories VALUES (1, 1, 'demo-1');
            INSERT INTO inventories VALUES (2, 2, 'demo-1');
            INSERT INTO inventory_parts
                VALUES (1, 'old-part', 5, 1, 'False', '');
            INSERT INTO inventory_parts
                VALUES (2, 'new-part', 5, 2, 'False', '');
            """
        )

        ingest.precompute(self.con)

        rows = self.con.execute(
            "SELECT set_num, part_num, color_id, quantity FROM set_parts"
        ).fetchall()
        self.assertEqual(rows, [("demo-1", "new-part", 5, 2)])

    def test_precompute_rebuilds_color_agnostic_index(self):
        self.con.executescript(
            """
            CREATE TABLE set_parts_any (
                set_num TEXT,
                part_num TEXT,
                quantity INTEGER
            );
            INSERT INTO set_parts_any VALUES ('stale-1', 'stale-part', 99);
            INSERT INTO inventories VALUES (1, 1, 'demo-1');
            INSERT INTO inventory_parts
                VALUES (1, 'brick', 5, 2, 'False', '');
            INSERT INTO inventory_parts
                VALUES (1, 'brick', 7, 3, 'False', '');
            """
        )

        ingest.precompute(self.con)

        rows = self.con.execute(
            "SELECT set_num, part_num, quantity FROM set_parts_any"
        ).fetchall()
        self.assertEqual(rows, [("demo-1", "brick", 5)])


class InventoryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = main.DB_PATH
        main.DB_PATH = str(Path(self.tmp.name) / "test.db")
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            con.executescript(
                """
                CREATE TABLE parts (part_num TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE colors (id INTEGER PRIMARY KEY, name TEXT, rgb TEXT);
                CREATE TABLE sets (
                    set_num TEXT PRIMARY KEY,
                    name TEXT,
                    year INTEGER,
                    theme_id INTEGER,
                    num_parts INTEGER,
                    img_url TEXT
                );
                CREATE TABLE set_parts (
                    set_num TEXT,
                    part_num TEXT,
                    color_id INTEGER,
                    quantity INTEGER
                );
                CREATE TABLE set_totals (
                    set_num TEXT PRIMARY KEY,
                    total_qty INTEGER,
                    distinct_parts INTEGER
                );
                CREATE TABLE my_parts (
                    part_num TEXT NOT NULL,
                    color_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL,
                    added_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (part_num, color_id)
                );
                INSERT INTO parts VALUES ('3001', 'Brick 2 x 4');
                INSERT INTO parts VALUES ('3020', 'Plate 2 x 4');
                INSERT INTO colors VALUES (5, 'Red', 'C91A09');
                INSERT INTO colors VALUES (7, 'Blue', '0055BF');
                INSERT INTO sets VALUES (
                    'demo-1', 'Serious Demo Set', 2026, 1, 5,
                    'https://example.test/demo.jpg'
                );
                INSERT INTO set_parts VALUES ('demo-1', '3001', 5, 2);
                INSERT INTO set_parts VALUES ('demo-1', '3020', 7, 3);
                INSERT INTO set_totals VALUES ('demo-1', 5, 2);
                """
            )

    def tearDown(self):
        main.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def test_inventory_rejects_non_positive_quantity(self):
        for quantity in (0, -3):
            with self.subTest(quantity=quantity):
                with self.assertRaises(ValidationError):
                    main.InvItem(
                        part_num="3001", color_id=5, quantity=quantity
                    )

    def test_inventory_rejects_unknown_color(self):
        with self.assertRaises(HTTPException) as caught:
            main.add_inventory(
                main.InvItem(part_num="3001", color_id=999, quantity=1)
            )

        self.assertEqual(caught.exception.status_code, 404)
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            count = con.execute("SELECT COUNT(*) FROM my_parts").fetchone()[0]
        self.assertEqual(count, 0)

    def test_inventory_still_accumulates_valid_items(self):
        item = main.InvItem(part_num="3001", color_id=5, quantity=2)
        main.add_inventory(item)
        main.add_inventory(item)

        with closing(sqlite3.connect(main.DB_PATH)) as con:
            quantity = con.execute(
                "SELECT quantity FROM my_parts WHERE part_num='3001' AND color_id=5"
            ).fetchone()[0]
        self.assertEqual(quantity, 4)

    def test_set_search_returns_import_preview(self):
        rows = main.search_sets(q="serious", limit=8)

        self.assertEqual(
            rows,
            [{
                "set_num": "demo-1",
                "name": "Serious Demo Set",
                "year": 2026,
                "num_parts": 5,
                "img_url": "https://example.test/demo.jpg",
                "total_qty": 5,
            }],
        )

    def test_import_owned_set_adds_complete_inventory_atomically(self):
        first = main.import_set_inventory("demo-1")
        second = main.import_set_inventory("demo-1")

        self.assertEqual(first["added_quantity"], 5)
        self.assertEqual(first["distinct_items"], 2)
        self.assertEqual(second["added_quantity"], 5)
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            rows = con.execute(
                """SELECT part_num, color_id, quantity FROM my_parts
                   ORDER BY part_num"""
            ).fetchall()
        self.assertEqual(rows, [("3001", 5, 4), ("3020", 7, 6)])

    def test_import_owned_set_rejects_unknown_set_without_changes(self):
        with self.assertRaises(HTTPException) as caught:
            main.import_set_inventory("missing-1")

        self.assertEqual(caught.exception.status_code, 404)
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            count = con.execute("SELECT COUNT(*) FROM my_parts").fetchone()[0]
        self.assertEqual(count, 0)


class ApiSchemaRegressionTests(unittest.TestCase):
    def test_bounded_query_parameters_have_lower_limits(self):
        main.app.openapi_schema = None
        paths = main.app.openapi()["paths"]

        buildable = {
            p["name"]: p["schema"]
            for p in paths["/api/buildable"]["get"]["parameters"]
        }
        search = {
            p["name"]: p["schema"]
            for p in paths["/api/parts/search"]["get"]["parameters"]
        }

        self.assertEqual(buildable["limit"]["minimum"], 1)
        self.assertEqual(search["limit"]["minimum"], 1)

    def test_scan_engine_is_restricted_to_supported_values(self):
        main.app.openapi_schema = None
        parameters = main.app.openapi()["paths"]["/api/scan"]["post"][
            "parameters"
        ]
        engine = next(p["schema"] for p in parameters if p["name"] == "engine")
        self.assertEqual(engine["pattern"], "^(fast|deep)$")


class LlmResponseRegressionTests(unittest.TestCase):
    def test_loose_json_parser_accepts_valid_json_with_trailing_commentary(self):
        text = 'Result: {"ideas": [{"title": "Duck"}]}\nHope this helps.'

        self.assertEqual(
            main.parse_json_loose(text),
            {"ideas": [{"title": "Duck"}]},
        )


class IdeaPreviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = main.DB_PATH
        main.DB_PATH = str(Path(self.tmp.name) / "test.db")
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            con.executescript(
                """
                CREATE TABLE sets (
                    set_num TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    year INTEGER,
                    num_parts INTEGER,
                    img_url TEXT
                );
                INSERT INTO sets VALUES (
                    '31152-1', 'Space Astronaut', 2024, 647,
                    'https://cdn.rebrickable.com/media/sets/31152-1.jpg'
                );
                """
            )

        image = Image.new("RGB", (1024, 1024), "#78C8F0")
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        self.image_bytes = encoded.getvalue()

    def tearDown(self):
        main.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    @patch("main._download_real_idea_preview", new_callable=AsyncMock)
    async def test_exact_real_set_image_is_cached_and_downloadable(self, download):
        download.return_value = (
            self.image_bytes,
            "image/jpeg",
            1024,
            1024,
        )
        idea = main.IdeaPreviewRequest(title="Space Astronaut")

        first = await main.create_idea_preview(idea)
        second = await main.create_idea_preview(idea)

        self.assertTrue(first["found"])
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["set_num"], "31152-1")
        self.assertEqual(first["set_name"], "Space Astronaut")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["width"], 1024)
        self.assertEqual(first["height"], 1024)
        download.assert_awaited_once_with(
            "https://cdn.rebrickable.com/media/sets/31152-1.jpg"
        )

        preview = main.get_idea_preview(first["id"])
        downloaded = main.download_idea_preview(first["id"])
        self.assertEqual(preview.body, self.image_bytes)
        self.assertEqual(preview.media_type, "image/jpeg")
        self.assertEqual(downloaded.body, self.image_bytes)
        self.assertIn(
            'attachment; filename="lego-set-31152-1.jpg"',
            downloaded.headers["content-disposition"],
        )

    @patch("main._download_real_idea_preview", new_callable=AsyncMock)
    async def test_original_moc_without_real_image_uses_placeholder(self, download):
        result = await main.create_idea_preview(
            main.IdeaPreviewRequest(title="Red and Blue Duck")
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["preview_url"], "/static/lego-placeholder.svg")
        self.assertIsNone(result["download_url"])
        download.assert_not_awaited()

    @patch("main._download_real_idea_preview", new_callable=AsyncMock)
    async def test_real_image_download_failure_falls_back_without_cache(self, download):
        download.side_effect = main.httpx.ConnectError("offline")
        result = await main.create_idea_preview(
            main.IdeaPreviewRequest(title="Space Astronaut")
        )

        self.assertFalse(result["found"])
        self.assertEqual(result["reason"], "real_image_unavailable")
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            count = con.execute("SELECT COUNT(*) FROM idea_previews").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_unrelated_claimed_set_number_is_not_used(self):
        result = await main.create_idea_preview(
            main.IdeaPreviewRequest(
                title="Tiny Rover", reference_set_num="31152-1"
            )
        )

        self.assertFalse(result["found"])

    async def test_untrusted_real_image_host_is_rejected_before_download(self):
        with self.assertRaises(ValueError):
            await main._download_real_idea_preview(
                "https://example.test/not-a-real-catalog-image.jpg"
            )

    def test_generated_preview_cache_is_removed_by_schema_migration(self):
        with closing(sqlite3.connect(main.DB_PATH)) as con:
            con.executescript(
                """
                CREATE TABLE idea_previews (
                    id TEXT PRIMARY KEY, title TEXT, prompt TEXT, image BLOB,
                    content_type TEXT, width INTEGER, height INTEGER,
                    created_at TEXT
                );
                INSERT INTO idea_previews VALUES (
                    'generated', 'Duck', 'AI prompt', X'00',
                    'image/jpeg', 768, 768, datetime('now')
                );
                """
            )
            main.ensure_idea_preview_table(con)
            columns = {
                row[1] for row in con.execute("PRAGMA table_info(idea_previews)")
            }
            count = con.execute("SELECT COUNT(*) FROM idea_previews").fetchone()[0]

        self.assertIn("source_url", columns)
        self.assertNotIn("prompt", columns)
        self.assertEqual(count, 0)


class StartupRegressionTests(unittest.TestCase):
    def test_application_import_does_not_depend_on_working_directory(self):
        root = Path(main.__file__).resolve().parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import main; "
                        "print(main.app.title); print(Path(main.DB_PATH).resolve())"
                    ),
                ],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines(),
            ["mylego-vision", str(root / "lego.db")],
        )


class FrontendSecurityRegressionTests(unittest.TestCase):
    def test_external_text_is_escaped_before_inner_html_rendering(self):
        html = (Path(main.__file__).resolve().parent / "static/index.html").read_text()

        self.assertIn("${escapeHtml(i.title || 'Untitled model')}", html)
        self.assertIn("${escapeHtml(i.description || '')}", html)
        self.assertIn("${escapeHtml(r.ollama.description || '')}", html)
        self.assertIn("${escapeHtml(c.name || '?')}", html)
        self.assertNotIn("${i.title}", html)
        self.assertNotIn("${i.description || ''}", html)
        self.assertNotIn("onclick=\"manualAdd('${", html)
        self.assertNotIn("onclick=\"toggleMissing(this,'${", html)

    def test_external_image_urls_are_protocol_filtered(self):
        html = (Path(main.__file__).resolve().parent / "static/index.html").read_text()

        self.assertIn("['http:', 'https:'].includes(url.protocol)", html)
        self.assertIn("safeImageUrl(i", html)

    def test_ai_ideas_use_real_images_or_local_placeholder_only(self):
        root = Path(main.__file__).resolve().parent
        sources = "\n".join(
            (root / path).read_text()
            for path in ("main.py", "static/index.html", "README.md")
        ).lower()

        self.assertNotIn("pollinations", sources)
        self.assertIn("/static/lego-placeholder.svg", sources)
        self.assertIn("cdn.rebrickable.com", sources)


if __name__ == "__main__":
    unittest.main()
