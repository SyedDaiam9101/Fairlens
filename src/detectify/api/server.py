"""Detectify FastAPI Server - REST API for object detection."""
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile  # pyright: ignore[reportMissingImports]
from fastapi.responses import HTMLResponse, Response, StreamingResponse  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel  # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]

from detectify import __version__
from detectify.config import settings
from detectify.db import (
    DetectionCreate,
    DetectionResponse,
    SourceType,
    create_tables,
    get_db,
    get_detections,
    save_detection,
)
from detectify.evaluation import draw_boxes, overlay_fps
from detectify.model.yolo import YOLODetector
from detectify.utils import list_cameras, logger


# Initialize database tables on startup
create_tables()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Detectify API starting...")
    logger.info(f"Database: {settings.database_url}")
    yield
    # Shutdown
    logger.info("Detectify API shutting down...")

# FastAPI app
app = FastAPI(
    title="Detectify API",
    description="YOLOv8 Object Detection REST API",
    version=__version__,
    lifespan=lifespan,
)

# Global model instance (lazy loaded, thread-safe)
_detector: Optional[YOLODetector] = None
_detector_lock = threading.Lock()


def get_detector() -> YOLODetector:
    """Get or initialize the detector model (thread-safe singleton)."""
    global _detector
    if _detector is None:
        with _detector_lock:
            # Double-check pattern to avoid race conditions
            if _detector is None:
                _detector = YOLODetector()
                # 'yolov8n.pt' is the NANO model - fastest available (30+ FPS on CPU)
                _detector.load('yolov8n.pt') 
    return _detector


# Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: dict[str, float]


class DetectResponse(BaseModel):
    detections: list[DetectionResult]
    count: int
    processing_time_ms: float


class PaginatedDetectionsResponse(BaseModel):
    detections: list[DetectionResponse]
    total: int
    limit: int
    offset: int


class CameraInfo(BaseModel):
    index: int
    name: str


class SensorPayload(BaseModel):
    timestamp: int
    pir1_triggered: bool
    pir2_triggered: bool
    distance_cm: float
    alert_level: str
    confidence_score: int
    image_data: str | None = None


# Endpoints
@app.post("/api/sensors/trigger")
async def trigger_sensor_alert(payload: SensorPayload):
    """
    Receive sensor triggers from ESP32.
    """
    # Log the event
    logger.info(f"🚨 SENSOR ALERT received from ESP32!")
    logger.info(f"   PIR1: {payload.pir1_triggered}, PIR2: {payload.pir2_triggered}")
    logger.info(f"   Distance: {payload.distance_cm} cm (Level: {payload.alert_level})")
    
    # Logic to handle the alert
    # For example, could set a global flag to start recording on local webcams
    # global recording_active
    # recording_active = True
    
    return {
        "status": "received", 
        "action": "alert_logged",
        "email_trigger": payload.alert_level == "critical", # Example logic
        "confidence_score": payload.confidence_score
    }


class CaptureRequest(BaseModel):
    action: str
    camera_id: int = 0
    reason: str | None = None


@app.post("/api/capture")
async def capture_from_webcam(request: CaptureRequest, db: Session = Depends(get_db)):
    """
    Capture an image from a connected USB webcam.
    Called by ESP32 when motion is detected.
    """
    logger.info(f"📷 Capture request from ESP32: camera={request.camera_id}, reason={request.reason}")
    
    try:
        # Open webcam
        cap = cv2.VideoCapture(request.camera_id)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail=f"Could not open camera {request.camera_id}")
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        
        # Run detection on captured frame
        detector = get_detector()
        detections = detector.predict(frame)
        
        # Draw boxes
        result_frame = draw_boxes(frame, detections)
        
        # Save to file
        timestamp = int(time.time())
        filename = f"esp32_capture_{timestamp}.jpg"
        filepath = f"crops/{filename}"
        cv2.imwrite(filepath, result_frame)
        
        logger.info(f"   Captured {len(detections)} detections, saved to {filepath}")
        
        # Save detections to database
        for det in detections:
            bbox = det.get("bbox", {})
            event = DetectionCreate(
                camera_id=str(request.camera_id),
                class_id=det.get("class_id", 0),
                class_name=det.get("class_name", "unknown"),
                confidence=det.get("confidence", 0.0),
                x1=bbox.get("x1", 0),
                y1=bbox.get("y1", 0),
                x2=bbox.get("x2", 0),
                y2=bbox.get("y2", 0),
                image_url=filepath,
                source_type=SourceType.WEBCAM,
            )
            save_detection(db, event)
        
        return {
            "status": "captured",
            "detections": len(detections),
            "image_path": filepath,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _detector
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=_detector is not None and _detector.is_loaded,
    )


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    format: str = Query("image", description="Response format: 'image' or 'json'"),
    db: Session = Depends(get_db),
):
    """
    Detect objects in an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        format: Response format - 'image' returns annotated image, 'json' returns detections
        
    Returns:
        Annotated image (JPEG) or JSON with detections.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Run detection
    detector = get_detector()
    start_time = time.time()
    detections = detector.predict(image)
    processing_time = (time.time() - start_time) * 1000  # (milliseconds)
    
    # Save to database
    for det in detections:
        bbox = det.get("bbox", {})
        
        # Save crop if enabled
        image_path = None
        if settings.save_crops:
            from detectify.utils import save_crop
            image_path = save_crop(
                image,
                bbox,
                det.get("class_name", "unknown"),
                settings.crop_dir,
            )

        event = DetectionCreate(
            camera_id=None,
            class_id=det.get("class_id", 0),
            class_name=det.get("class_name", "unknown"),
            confidence=det.get("confidence", 0.0),
            x1=bbox.get("x1", 0),
            y1=bbox.get("y1", 0),
            x2=bbox.get("x2", 0),
            y2=bbox.get("y2", 0),
            image_url=image_path,
            source_type=SourceType.IMAGE,
        )
        save_detection(db, event)
    
    if format == "json":
        return DetectResponse(
            detections=[
                DetectionResult(
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                )
                for d in detections
            ],
            count=len(detections),
            processing_time_ms=processing_time,
        )
    
    # Draw boxes and return image
    result_image = draw_boxes(image, detections)
    
    # Encode to JPEG
    _, encoded = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"X-Detection-Count": str(len(detections))},
    )


@app.get("/detect/live")
async def detect_live(
    camera: str = Query("auto", description="Camera index or 'auto'"),
):
    """
    Stream live detection via MJPEG.
    
    Args:
        camera: Camera index (0, 1, ...) or 'auto' to select first available.
        
    Returns:
        MJPEG video stream with detections.
    """
    # Determine camera source
    if camera == "auto":
        cameras = list_cameras()
        if not cameras:
            raise HTTPException(status_code=503, detail="No cameras available")
        source = cameras[0][0]
    else:
        # Try to parse as int (local cam index), otherwise treat as URL string
        try:
            source = int(camera)
        except ValueError:
            source = camera
    
    def generate_frames():
        """Generator that yields MJPEG frames."""
        detector = get_detector()
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail=f"Could not open camera {source}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                detections = detector.predict(frame)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw boxes
                result_frame = draw_boxes(frame, detections)
                result_frame = overlay_fps(result_frame, fps)
                
                # Encode frame
                _, encoded = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections", response_model=PaginatedDetectionsResponse)
async def get_detection_events(
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    class_name: Optional[str] = Query(None, description="Filter by class name"),
    start_ts: Optional[datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_ts: Optional[datetime] = Query(None, description="End timestamp (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Get paginated list of detection events from database.
    
    Supports filtering by camera_id, class_name, and time range.
    """
    detections, total = get_detections(
        db,
        camera_id=camera_id,
        class_name=class_name,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
        offset=offset,
    )
    
    return PaginatedDetectionsResponse(
        detections=[
            DetectionResponse.from_orm(d)
            for d in detections
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/cameras", response_model=list[CameraInfo])
async def list_available_cameras():
    """List available camera devices."""
    cameras = list_cameras()
    return [CameraInfo(index=idx, name=name) for idx, name in cameras]


@app.get("/export/csv")
async def export_detections_csv(db: Session = Depends(get_db)):
    """Download all detections as a CSV file (Excel compatible)."""
    import csv
    import io
    
    # Query all detections
    detections, _ = get_detections(db, limit=10000) # Export last 10000 events
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "ID", "Timestamp", "Camera ID", "Class Name", 
        "Confidence", "Is Unauthorised", "Source", "Image URL"
    ])
    
    # Rows
    for d in detections:
        writer.writerow([
            d.id,
            d.timestamp.isoformat() if d.timestamp else "",
            d.camera_id or "",
            d.class_name,
            f"{d.confidence:.2f}",
            "Yes" if getattr(d, 'unauthorised', False) else "No",
            d.source_type.value if d.source_type else "",
            d.image_url or ""
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=detections_{int(time.time())}.csv"}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    limit: int = Query(50, ge=1, le=500, description="Number of detections to show"),
    db: Session = Depends(get_db),
):
    """
    Web dashboard to view detection history.
    
    Access at: http://localhost:8001/dashboard
    """
    try:
        detections, total = get_detections(db, limit=limit, offset=0)
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Sentinel Dashboard - Error</title>
                <style>
                    body {{ font-family: sans-serif; background: #121212; color: #eee; padding: 40px; text-align: center; }}
                    .error {{ background: #ff4444; padding: 20px; border-radius: 8px; margin: 20px auto; max-width: 600px; }}
                </style>
            </head>
            <body>
                <h1>🏔️ Sentinel Dashboard</h1>
                <div class="error">
                    <h2>Error Loading Dashboard</h2>
                    <p>{str(e)}</p>
                    <p>Please check the server logs for details.</p>
                    <p><a href="/health" style="color: #fff;">Check Health</a> | <a href="/docs" style="color: #fff;">API Docs</a></p>
                </div>
            </body>
        </html>
        """
    
    table_rows = ""
    for d in detections:
        try:
            img_html = (
                f'<a href="/{d.image_url}" target="_blank"><img src="/{d.image_url}" height="60" style="border-radius: 4px;"></a>'
                if d.image_url
                else '<span style="color: #888;">No Image</span>'
            )
            # Safely access unauthorised field (may not exist in older DB records)
            is_unauthorised = getattr(d, 'unauthorised', False)
            unauth_badge = (
                '<span style="background: #ff4444; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">UNAUTHORIZED</span>'
                if is_unauthorised
                else '<span style="background: #44ff44; color: #000; padding: 2px 8px; border-radius: 3px; font-size: 11px;">AUTHORIZED</span>'
            )
            conf_color = "#ff6666" if d.confidence < 0.5 else "#66ff66" if d.confidence > 0.8 else "#ffaa66"
            
            table_rows += f"""
            <tr>
                <td>{d.timestamp.strftime('%Y-%m-%d %H:%M:%S') if d.timestamp else 'N/A'}</td>
                <td>{d.camera_id or 'N/A'}</td>
                <td><strong>{d.class_name}</strong></td>
                <td style="color: {conf_color}; font-weight: bold;">{(d.confidence * 100):.1f}%</td>
                <td>{unauth_badge}</td>
                <td>{d.source_type.value if d.source_type else 'N/A'}</td>
                <td>{img_html}</td>
            </tr>
            """
        except Exception as e:
            logger.error(f"Error processing detection {d.id}: {e}")
            continue
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Detectify Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #1e1e2e 0%, #121212 100%);
                    color: #e0e0e0;
                    padding: 20px;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                header {{
                    background: rgba(255, 255, 255, 0.05);
                    padding: 20px 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                h1 {{
                    color: #00ff88;
                    font-size: 2em;
                    margin-bottom: 5px;
                }}
                .subtitle {{
                    color: #aaa;
                    font-size: 0.9em;
                }}
                .header-actions {{
                    display: flex;
                    gap: 10px;
                }}
                .action-btn {{
                    background: #00ff88;
                    color: #000;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                    transition: transform 0.2s;
                }}
                .action-btn:hover {{
                    transform: scale(1.05);
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: rgba(255, 255, 255, 0.05);
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #00ff88;
                }}
                .stat-label {{
                    color: #aaa;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                thead {{
                    background: rgba(0, 255, 136, 0.1);
                }}
                th {{
                    padding: 15px;
                    text-align: left;
                    color: #00ff88;
                    font-weight: 600;
                    border-bottom: 2px solid rgba(0, 255, 136, 0.3);
                }}
                td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }}
                tr:hover {{
                    background: rgba(255, 255, 255, 0.05);
                }}
                .links {{
                    margin-top: 20px;
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                }}
                .link-btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: rgba(0, 255, 136, 0.2);
                    color: #00ff88;
                    text-decoration: none;
                    border-radius: 5px;
                    border: 1px solid rgba(0, 255, 136, 0.3);
                    transition: all 0.3s;
                }}
                .link-btn:hover {{
                    background: rgba(0, 255, 136, 0.3);
                    transform: translateY(-2px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div>
                        <h1>🔍 Detectify Dashboard</h1>
                        <div class="subtitle">Real-time Object Detection Monitoring System</div>
                    </div>
                    <div class="header-actions">
                        <a href="/export/csv" class="action-btn">📥 Download Excel (CSV)</a>
                    </div>
                </header>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Detections</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len([d for d in detections if getattr(d, 'unauthorised', False)])}</div>
                        <div class="stat-label">Unauthorized Alerts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(set(d.camera_id for d in detections if d.camera_id))}</div>
                        <div class="stat-label">Active Cameras</div>
                    </div>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Camera ID</th>
                            <th>Class</th>
                            <th>Confidence</th>
                            <th>Status</th>
                            <th>Source</th>
                            <th>Image</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows if table_rows else '<tr><td colspan="7" style="text-align: center; padding: 40px; color: #888;">No detections found</td></tr>'}
                    </tbody>
                </table>
                
                <div class="links">
                    <a href="/docs" class="link-btn">📚 API Documentation</a>
                    <a href="/detect/live" class="link-btn">📹 Live Stream</a>
                    <a href="/health" class="link-btn">💚 Health Check</a>
                    <a href="/detections?limit=100" class="link-btn">📊 JSON API</a>
                </div>
            </div>
        </body>
    </html>
    """



