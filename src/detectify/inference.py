"""Detectify Inference Engine - Image, video, and webcam detection."""
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from detectify.config import settings
from detectify.db import DetectionCreate, SourceType, get_db_session, save_detection
from detectify.evaluation import draw_boxes, overlay_fps, overlay_detection_count
from detectify.model.yolo import YOLODetector
from detectify.utils import ask_camera, ensure_dir, logger


class InferenceEngine:
    """Unified inference engine for images, videos, and webcams."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the inference engine."""
        self.model = YOLODetector()
        # 'yolov8n.pt' is defaults to Nano (Fastest) in YOLODetector
        self.model.load(model_path or 'yolov8n.pt')
        self.save_video = False
        self.writer = None

    def load_model(self) -> "InferenceEngine":
        """Load the detection model."""
        # Model is loaded in __init__ for YOLODetector
        # This method can be kept for compatibility or removed if not needed elsewhere.
        logger.info("Model already loaded during initialization.")
        return self

    def process_image(
        self,
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        show: bool = False,
        save_to_db: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to input image.
            output_path: Optional path to save result image.
            show: Whether to display result in window.
            save_to_db: Whether to log detections to database.
            
        Returns:
            List of detections.
        """
        self.load_model()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Run detection
        detections = self.model.predict(image)
        logger.info(f"Found {len(detections)} objects")
        
        # Save to database
        if save_to_db:
            self._save_detections(
                detections,
                source_type=SourceType.IMAGE,
                camera_id=None,
                image=image,
            )
        
        # Draw boxes on image
        result_image = draw_boxes(image, detections)
        result_image = overlay_detection_count(result_image, len(detections))
        
        # Save output image
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"Saved result to: {output_path}")
        
        # Show result
        if show:
            cv2.imshow("Detectify", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        show: bool = True,
        save_to_db: bool = True,
    ) -> dict[str, Any]:
        """
        Run detection on a video file.
        
        Args:
            video_path: Path to input video.
            output_path: Optional path to save result video.
            show: Whether to display frames in window.
            save_to_db: Whether to log detections to database.
            
        Returns:
            Statistics dict with total frames, detections, fps.
        """
        self.load_model()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames, {fps:.1f} FPS)")
        
        # Setup video writer
        writer = None
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                detections = self.model.predict(frame)
                total_detections += len(detections)
                
                # Save to database (sample every N frames to avoid overload)
                if save_to_db and frame_count % 10 == 0:
                    self._save_detections(
                        detections,
                        source_type=SourceType.VIDEO,
                        camera_id=None,
                        image=frame,
                    )
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw on frame
                result_frame = draw_boxes(frame, detections)
                result_frame = overlay_fps(result_frame, current_fps)
                result_frame = overlay_detection_count(result_frame, len(detections))
                
                # Write to output
                if writer:
                    writer.write(result_frame)
                
                # Display
                if show:
                    cv2.imshow("Detectify", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                # Progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"Completed: {frame_count} frames, {total_detections} detections, {avg_fps:.1f} FPS")
        
        return {
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "avg_fps": avg_fps,
            "duration_seconds": elapsed,
        }

    def process_webcam(
        self,
        camera_id: Optional[int | str] = None,
        save_to_db: bool = True,
    ) -> dict[str, Any]:
        """
        Run live detection on webcam.
        
        Args:
            camera_id: Camera device index. If None, prompts user.
            save_to_db: Whether to log detections to database.
            
        Returns:
            Statistics dict.
        """
        self.load_model()
        
        # Get camera selection
        if camera_id is None:
            camera_id = ask_camera()
        
        logger.info(f"Opening camera {camera_id}")
        
        if sys.platform == "win32":
            # DirectShow is generally faster/more reliable for webcam enumeration and access
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default/MSMF
                cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        logger.info("Starting live detection. Press 'q' to quit.")
        
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        last_db_save = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                frame_count += 1
                
                # Run detection
                detections = self.model.predict(frame)
                total_detections += len(detections)
                
                # Save to database (rate limited)
                if save_to_db and time.time() - last_db_save > 1.0:
                    self._save_detections(
                        detections,
                        source_type=SourceType.WEBCAM,
                        camera_id=camera_id,
                        image=frame,
                    )
                    last_db_save = time.time()
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw on frame
                result_frame = draw_boxes(frame, detections)
                result_frame = overlay_fps(result_frame, current_fps)
                result_frame = overlay_detection_count(result_frame, len(detections))
                
                # Display
                cv2.imshow("Detectify - Live", result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, result_frame)
                    logger.info(f"Saved screenshot: {screenshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"Session ended: {frame_count} frames, {total_detections} detections")
        
        return {
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "avg_fps": avg_fps,
            "duration_seconds": elapsed,
            "camera_id": camera_id,
        }

    def _save_detections(
        self,
        detections: list[dict[str, Any]],
        source_type: SourceType,
        camera_id: Optional[int | str],
        image: Optional[np.ndarray] = None,
    ) -> None:
        """Save detections to database."""
        if not detections:
            return
        
        with get_db_session() as db:
            for det in detections:
                bbox = det.get("bbox", {})
                
                # Save crop if enabled
                image_path = None
                if settings.save_crops and image is not None:
                    from detectify.utils import save_crop
                    image_path = save_crop(
                        image,
                        bbox,
                        det.get("class_name", "unknown"),
                        settings.crop_dir,
                    )

                # Database support for both integer and string camera_id
                db_camera_id = str(camera_id) if camera_id is not None else None

                # Calculate unauthorised status
                conf = det.get("confidence", 0.0)
                is_person = det.get("class_name") == "person"
                # threshold from settings
                threshold = settings.confidence_threshold
                unauthorised = is_person and (conf * 100 < threshold)

                event = DetectionCreate(
                    camera_id=db_camera_id,
                    class_id=det.get("class_id", 0),
                    class_name=det.get("class_name", "unknown"),
                    confidence=conf,
                    x1=bbox.get("x1", 0),
                    y1=bbox.get("y1", 0),
                    x2=bbox.get("x2", 0),
                    y2=bbox.get("y2", 0),
                    person_detected=is_person,
                    unauthorised=unauthorised,
                    image_url=image_path,  # Now called image_url in DB
                    source_type=source_type,
                    # These would come from sensor fusion in a real scenario
                    distance_cm=None,
                    motion=False, 
                )
                save_detection(db, event)

                # Trigger Email Alert for "unauthorised person"
                if unauthorised and settings.enable_notifications:
                    from detectify.utils.notifications import notifier
                    notifier.send_alert(
                        subject=f"⚠️ UNAUTHORISED PERSON DETECTED (Conf: {conf*100:.1f}%)",
                        message=f"A person was detected with low confidence: {conf*100:.1f}%. Treat as intruder.",
                        image_path=image_path,
                    )

                # Trigger IoT Alert (ESP32)
                if det.get("class_name") == "person" and settings.enable_iot:
                    from detectify.utils.iot import iot_manager
                    iot_manager.trigger_alert(
                        class_name="person",
                        confidence=det["confidence"],
                    )


def run_inference(
    source: Optional[str] = None,
    output: Optional[str] = None,
    show: bool = True,
    save_to_db: bool = True,
) -> dict[str, Any]:
    """
    Main inference function.
    
    Args:
        source: Path to image/video or None for webcam.
        output: Optional output path.
        show: Whether to display results.
        save_to_db: Whether to save detections to database.
        
    Returns:
        Results dictionary.
    """
    engine = InferenceEngine()
    
    if source is None:
        # Webcam mode
        return engine.process_webcam(save_to_db=save_to_db)
    
    source_path = Path(source)
    
    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # Image mode
        detections = engine.process_image(
            source_path,
            output_path=output,
            show=show,
            save_to_db=save_to_db,
        )
        return {"detections": detections, "count": len(detections)}
    
    elif source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        # Video mode
        return engine.process_video(
            source_path,
            output_path=output,
            show=show,
            save_to_db=save_to_db,
        )
    
    elif str(source).startswith(("http://", "https://", "rtsp://")):
        # Remote stream mode (ESP32)
        return engine.process_video(
            source,
            output_path=output,
            show=show,
            save_to_db=save_to_db,
        )
    
    else:
        raise ValueError(f"Unsupported source format: {source}")
