# mylego · vision

What can I build from my pile of LEGO parts?
Scan parts with computer vision → keep an inventory → rank every real LEGO set
by buildability → ask a local LLM for creative MOC ideas.

## Architecture

```
photo --> Brickognize API (part id, free CV service trained on all LEGO parts)
      --> local dominant-color -> nearest LEGO color (Rebrickable palette)
      --> optional: Ollama vision LLM (qwen3.6:27b on the GPU box) second opinion
                                    │
                                    ▼
        SQLite (lego.db) <-- Rebrickable full dataset (~27k sets, 63k parts,
                             1.5M inventory rows, precomputed set_parts)
                                    │
        my_parts inventory ---------┤
                                    ▼
        buildability engine: SUM(MIN(need, have)) / SUM(need) per set
        (strict = exact color match, loose = shape only)
                                    │
                                    ▼
        AI advisor: Ollama text LLM invents MOCs from your actual parts
```

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 ingest.py                     # downloads Rebrickable dumps -> lego.db (~1 min)
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8500
# open http://localhost:8500
```

Config in `.env`:

```bash
OLLAMA_HOST=http://10.10.10.100:11434     # your GPU box address (Nvidia)
OLLAMA_VISION_MODEL=qwen3.6:27b        # any vision-capable ollama model
OLLAMA_TEXT_MODEL=qwen3.6:27b
```

GPU vs CPU: all heavy AI runs on the Ollama host; this app itself is
CPU-only (SQLite + tiny Pillow color math) and runs anywhere. To go
fully-CPU later just point `OLLAMA_HOST` at a smaller local model
(e.g. `qwen3:4b` + a small vision model) - nothing else changes.

## Workflow

1. **Scan** - one part on a white background (start simple: 1 part → 5-10 laid
   out → the full pile later). `fast` = Brickognize + color detect,
   `deep` = also asks the Ollama vision model.
2. **Inventory** - confirmed parts land in `my_parts` (part_num + color + qty).
   Manual add by part number/name also works.
3. **Builds** - every real LEGO set ranked by coverage: 100% buildable,
   90%+ near-buildable, etc. Click a set to see which parts are missing.
   `exact color` / `any color` modes.
4. **AI Ideas** - the local LLM suggests original small MOCs using only
   your parts.

Sample part photos to try: `test_images/*.jpg`.

## API

| endpoint | what |
|---|---|
| `POST /api/scan?engine=fast\|deep` | identify a part on a photo |
| `GET/POST/DELETE /api/inventory` | my parts CRUD |
| `GET /api/buildable?mode=strict\|loose` | ranked buildable sets |
| `GET /api/buildable/{set}/missing` | missing-parts diff for a set |
| `POST /api/advise` | LLM MOC ideas from inventory |
| `GET /api/status` | db + ollama health |

## Data & credits

- [Rebrickable](https://rebrickable.com/downloads/) - full LEGO catalog CSV dumps (CC-BY-4.0-ish, see their terms)
- [Brickognize](https://brickognize.com) - free part-recognition API by Piotr Rybak
- next steps: YOLO fine-tune on [B200C LEGO renders dataset](https://mostwiedzy.pl/en/open-research-data/b200c-lego-classification-dataset-200-bricks-800000-images,209111855855869-0)
  for multi-part pile detection, LDraw/LeoCAD for build instructions.
