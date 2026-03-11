# Detectify

Production-ready YOLOv8-based Object Detection framework with FastAPI, SQLAlchemy (SQLite/PostgreSQL), and live webcam support.

## 🚀 Key Features

- **YOLOv8 Backend**: Powered by Ultralytics YOLOv8 (default: YOLOv8n).
- **Unified Inference**: Process images, video files, or live webcam feeds.
- **FastAPI Core**: High-performance REST API with MJPEG live streaming.
- **Persistent Audit**: Every detection is automatically logged to SQLite/PostgreSQL.
- **DB History**: Paginated and filterable query endpoint for detection events.
- **Production Infrastructure**: Multi-stage Docker optimization and GitHub Actions CI.

## 🛠 Installation

### 1. Prerequisites
- Python 3.10+
- OpenCV system dependencies (`libgl1-mesa-glx`, etc.)
- (Optional) CUDA-enabled GPU for acceleration.

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/detectify.git
cd detectify

# Install using Makefile
make install
```

### 3. Configuration
Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

## 🖥 Usage

### CLI Inference
```bash
# Live Webcam (Interactive camera selector)
python -m detectify inference

# Specific camera index
python -m detectify inference --camera 0

# Static Image
python -m detectify inference --source path/to/image.jpg --output output.jpg

# Video File
python -m detectify inference --source path/to/video.mp4 --output output.mp4
```

### API Server
```bash
# Start the FastAPI server
make serve

# Health Check
curl http://localhost:8000/health

# Post Detection (Annotated Image)
curl -X POST -F "file=@image.jpg" http://localhost:8000/detect --output result.jpg

# Post Detection (JSON)
curl -X POST -F "file=@image.jpg" "http://localhost:8000/detect?format=json"

# Query History
curl "http://localhost:8000/detections?class_name=person&limit=10"
```

## 🏗 Development

- **Initialize/Reset DB**: `make init-db`
- **Run Tests**: `make test`
- **Linting**: `make lint`
- **Build Docker**: `make docker-build`

## 📂 Project Structure
```text
detectify/
├── src/detectify/
│   ├── api/         # FastAPI endpoints & server
│   ├── db/          # SQLAlchemy models & CRUD
│   ├── evaluation/  # Visualizers & Metrics (mAP)
│   ├── model/       # YOLOv8 model wrappers
│   └── utils/       # Helpers & Logger
├── configs/         # YAML configurations
├── alembic/         # Database migrations
└── scripts/         # Utility shell/python scripts
```
