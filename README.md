# Nigerian Vehicle Number Plate Recognition

A modular VNPR/ALPR system for Nigerian vehicle plates. The pipeline detects plates with YOLO and recognizes characters with OCR, exposing both a FastAPI endpoint and a small browser client.

## Features

- YOLO plate detection through Ultralytics (`yolov8n.pt` or a custom plate model)
- EasyOCR for character recognition with lazy model loading
- Nigerian plate text cleanup and confidence-aware result objects
- Optional vehicle make/model classification through a configured YOLO classifier
- SQLite recognition history for every detected plate
- Admin signup/login with protected dashboard sessions
- Image upload API and a browser UI
- Unit tests that run without downloading a model

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. Create the first admin account from the signup tab, then sign in to use recognition and history. Set `AUTH_SECRET` to a long random value outside development; it signs the HTTP-only dashboard session cookie. Set `YOLO_MODEL_PATH` to a trained plate detector before using real images. A generic COCO YOLO model is useful for wiring checks but is not trained to detect plates.

The included `models/vbnpr_model.pt` is automatically used for vehicle make detection. Its classes identify makes such as Toyota, Honda, and Ford, plus `Plate_Number`; it does not identify individual vehicle models, so the API's `model` field remains `null`. To use another vehicle model, set `VEHICLE_MODEL_PATH` to a YOLO classification or detection model. Classification labels can use the `make|model` format, such as `Toyota|Corolla`.

Recognition history is stored in `data/vnpr.sqlite3` by default. Set `DATABASE_PATH` to change the database location. Retrieve the newest records with `GET /api/v1/history?limit=100`.

Successful valid recognitions are also persisted as one vehicle profile per normalized plate in `vehicles`, with every upload or live-camera event stored in `vehicle_recognition_history`. Use the protected `GET /api/v1/vehicles` endpoint for profiles and `GET /api/v1/vehicle-history?plate_number=ABC123AB` for searchable event history. Live recognition records once per tracker lifecycle; `TRACK_TIMEOUT` controls when a vehicle is considered gone and can create a new event when it returns.

Detected plates are checked against the administrator-managed `registered_plates` table. Recognition responses and history include `is_registered`, `is_verified`, and `registration_status` (`registered_verified`, `registered_unverified`, or `not_registered_unverified`). Use the protected registry endpoints to maintain it: `POST /api/v1/registry` with `{"plate_text":"ABC-123AB","is_verified":true}`, `GET /api/v1/registry`, and `DELETE /api/v1/registry/{plate_text}`.

Admin accounts are stored in the same SQLite database in the `admins` table. Passwords are stored as salted `scrypt` hashes, never as plain text. Use `SESSION_DAYS` to control session duration.

Run tests with:

```powershell
python -m pytest
```

## Detection diagnostics

Run the detector-only diagnostic against an image without invoking OCR:

```powershell
python test_detection.py path\to\vehicle.jpg
```

The command prints the model path, class names, image dimensions, confidence and image-size attempts, tiled/fallback counts, and the selected bounding box. It writes `debug_detection.jpg` with the selected box; green means YOLO and orange means the OpenCV fallback. The API also exposes the same information in the additive `detection` field returned by `POST /api/v1/recognize`, and `POST /api/v1/debug/detect` returns the complete threshold/size matrix.

Detection defaults are configurable with `YOLO_CONFIDENCE`, `YOLO_IOU`, `YOLO_IMGSZ`, `YOLO_MULTISCALE`, and `YOLO_TILE_OVERLAP`. The robust pipeline tries the full image at 640, 832, 1024, and the configured size before overlapping tiles and the conservative OpenCV candidate generator.

## Training a plate detector

Prepare YOLO labels with one class named `license_plate`, then create a dataset YAML:

```yaml
path: data/plates
train: images/train
val: images/val
names:
  0: license_plate
```

Train with `yolo detect train data=data/plates.yaml model=yolov8n.pt epochs=100 imgsz=640`. Put the resulting weights at `models/license_plate.pt` or update `.env`.

## API

`POST /api/v1/recognize` accepts a multipart image field named `file` and returns detections, plate text, confidence, and bounding boxes. `GET /health` reports whether the configured model and OCR engines are available.

This is a research system: validate predictions against a labelled Nigerian dataset before using it for enforcement or other high-impact decisions.