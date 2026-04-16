# Smart Match

Smart Match extracts structured genealogical records from scanned metrical book pages (birth, marriage, and death records) using a multi-stage OCR pipeline. The output is structured JSON with per-field confidence scores and a human-review flag.

## Quick Start

### 1. Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in the **required** fields before starting:

#### Handwritten OCR models — Hugging Face IDs `[REQUIRED]`

The stack uses TrOCR for handwritten text. Without local fine-tuned checkpoints the service pulls base models from Hugging Face on first start. Set these four fields to valid Hugging Face model IDs:

```env
SMARTMATCH_TROCR_BASE_MODEL_KAZARS=kazars24/trocr-base-handwritten-ru
SMARTMATCH_TROCR_BASE_PROCESSOR_KAZARS=kazars24/trocr-base-handwritten-ru
SMARTMATCH_TROCR_BASE_MODEL_CYRILLIC=cyrillic-trocr/trocr-handwritten-cyrillic
SMARTMATCH_TROCR_BASE_PROCESSOR_CYRILLIC=kazars24/trocr-base-handwritten-ru
```

The container must have internet access on the first start so the models can be downloaded and cached.

#### If you have locally fine-tuned TrOCR checkpoints `[OPTIONAL]`

Place the checkpoint directories under `./models/trocr/` (one subdirectory per model) and set:

```env
SMARTMATCH_TROCR_MODELS=/workspace/models/trocr/kazars:/workspace/models/trocr/cyrillic
```

Leave `SMARTMATCH_TROCR_MODELS` empty to skip local weights and use only the Hugging Face base models.

#### Classifier weights

The pre-trained handwritten/typed classifier is shipped in `./models/classifier.pth`. No changes needed unless you retrain it and save it to a different path.

### 2. Start the stack

```bash
docker compose up --build
```

| Service | Default URL |
|---|---|
| Frontend | http://localhost:9999 |
| OCR API | http://localhost:8000 |
| Training API | http://localhost:8100 |

Ports are controlled by `.env` and can be changed freely.

---

## Architecture

The system is split into three independently deployable Docker services coordinated by Docker Compose.

```
┌─────────────────────────────────────────┐
│            smartmatch-frontend          │
│  nginx — serves static UI, proxies      │
│  /api/ocr/* → smartmatch-ocr:8000       │
│  /api/training/* → smartmatch-training  │
└────────────┬────────────────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│   OCR   │    │   Training   │
│  :8000  │    │    :8100     │
└─────────┘    └──────────────┘
```

### OCR Service (`SmartMatch/OCRContainer`)

Runs the five-stage document extraction pipeline:

| Stage | Implementation | Notes |
|---|---|---|
| **Preprocess** | Kraken binarization | deskew, binarize |
| **Segment** | OpenCV (projection + Hough) | splits page into column strips |
| **Classify** | SimpleCNN (`classifier.pth`) | printed vs handwritten per segment |
| **OCR** | PaddleOCR (printed) / TrOCR (handwritten) | two specialized engines |
| **NER** | Natasha | persons, locations, dates |

The production endpoint `POST /api/ocr/process` runs all five stages and returns structured JSON. Individual stages can be called independently via `POST /api/ocr/stages/{stage}` or `POST /api/ocr/run` for debugging.

### Training Service (`SmartMatch/TrainingService` + `SmartMatch/TrainingContainer`)

The Training Service is a thin FastAPI wrapper that launches training scripts from `TrainingContainer` as subprocesses and streams their logs. Two models can be trained:

- **TrOCR (kazars / cyrillic)** — fine-tunes `VisionEncoderDecoderModel` on `OCR/joined_data` (image–text pairs).
- **Handwritten/Typed Classifier** — trains the SimpleCNN on a labeled image directory.

Job status and logs are polled from the frontend.

### Frontend (`SmartMatch/OCRContainer/Backend/frontend`)

A static nginx application. Provides:
- OCR console (stage selection, file upload, structured result view)
- Training console (model selection, hyperparameter form, live job log)
- Links to Swagger UI for both APIs

---

## OCR Output Schema

`POST /api/ocr/process` returns:

```json
{
  "record_type": "birth",
  "record_type_confidence": 82,
  "extracted_fields": {
    "child_name":    { "value": "Ivan Petrov",   "confidence": 76, "low_confidence": false, "source": "ner/person-order" },
    "birth_date":    { "value": "1891-03-14",     "confidence": 81, "low_confidence": false, "source": "date-extractor"   },
    "baptism_date":  { "value": "1891-03-16",     "confidence": 81, "low_confidence": false, "source": "date-extractor"   },
    "parents_names": { "value": ["Pyotr Petrov"], "confidence": 56, "low_confidence": true,  "source": "ner/person-order" },
    "godparents":    { "value": [],               "confidence": 56, "low_confidence": true,  "source": "ner/person-order" },
    "location":      { "value": "Kazan",          "confidence": 64, "low_confidence": false, "source": "ner/location"     }
  },
  "human_review_required": false,
  "document_metadata": {
    "filename": "metrical_book_1891.jpg",
    "processing_time": 4.2,
    "run_id": "ocr_123456789abc",
    "ocr_confidence": 81,
    "human_review_required": false,
    "selected_stages": ["preprocess", "segment", "classify", "ocr", "ner"],
    "stage_summary_url": "/api/ocr/runs/ocr_123456789abc/summary"
  }
}
```

Record types: `birth`, `marriage`, `death`, `unknown`. Fields vary by type; see `document_schema.py` for the full field list per type.

---

## Key API Endpoints

### OCR Service

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/ocr/config` | Pipeline metadata (stages, models, allowed extensions) |
| POST | `/api/ocr/process` | Full pipeline → structured JSON |
| POST | `/api/ocr/run` | Run a stage slice (debug) |
| POST | `/api/ocr/stages/{stage}` | Run a single stage |
| GET | `/api/ocr/runs/{run_id}/summary` | Full run summary |
| GET | `/docs` | Swagger UI |

### Training Service

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/training/models` | Available trainable models |
| GET | `/api/training/jobs` | All training jobs |
| POST | `/api/training/jobs` | Start a training job |
| GET | `/api/training/jobs/{id}` | Job status |
| GET | `/api/training/jobs/{id}/log` | Training log |
| GET | `/docs` | Swagger UI |

---

## CLI

The OCR pipeline can also be run from the command line inside the container:

```bash
python SmartMatch/OCRContainer/Backend/cli.py \
  --input path/to/image.jpg \
  --output path/to/output.json
```

Add `--include-debug` to embed the stage-by-stage payload in the output file.
