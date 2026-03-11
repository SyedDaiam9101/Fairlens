"""Detectify CLI - Command-line interface entry point."""
import argparse
import sys

from detectify import __version__
from detectify.config import settings
from detectify.db import create_tables
from detectify.utils import logger


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="detectify",
        description="Detectify - TensorFlow Object Detection",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"detectify {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Inference command
    infer_parser = subparsers.add_parser("inference", help="Run object detection")
    infer_parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Path to image or video. If not provided, opens webcam.",
    )
    infer_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output image/video.",
    )
    infer_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display results in window.",
    )
    infer_parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't save detections to database.",
    )
    infer_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index for webcam mode.",
    )
    infer_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to custom model weights (e.g., best.pt).",
    )
    
    # Serve command (API server)
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument(
        "--host",
        type=str,
        default=settings.api_host,
        help=f"Host to bind to (default: {settings.api_host})",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help=f"Port to bind to (default: {settings.api_port})",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    
    # Init-db command
    subparsers.add_parser("init-db", help="Initialize database tables")
    
    # IoT Test command
    subparsers.add_parser("iot-test", help="Test connection to ESP32 module")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "config",
        nargs="?",
        default="data.yaml",
        help="Path to training config or data.yaml (default: data.yaml)",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "inference":
        from detectify.inference import InferenceEngine
        
        # Initialize database
        create_tables()
        
        engine = InferenceEngine(model_path=args.model)
        
        if args.source:
            from pathlib import Path
            source_path = Path(args.source)
            
            if args.source.isdigit():
                # Treat "0", "1" as webcam index
                engine.process_webcam(
                    camera_id=int(args.source),
                    save_to_db=not args.no_db,
                )
            elif source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                engine.process_image(
                    args.source,
                    output_path=args.output,
                    show=not args.no_show,
                    save_to_db=not args.no_db,
                )
            elif str(args.source).startswith(("http://", "https://", "rtsp://")):
                # Remote stream (ESP32)
                engine.process_video(
                    args.source,
                    output_path=args.output,
                    show=not args.no_show,
                    save_to_db=not args.no_db,
                )
            else:
                engine.process_video(
                    args.source,
                    output_path=args.output,
                    show=not args.no_show,
                    save_to_db=not args.no_db,
                )
        else:
            # Webcam mode
            engine.process_webcam(
                camera_id=args.camera,
                save_to_db=not args.no_db,
            )
        
        return 0
    
    elif args.command == "serve":
        import uvicorn  # pyright: ignore[reportMissingImports]
        
        # Initialize database
        create_tables()
        
        uvicorn.run(
            "detectify.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0
    
    elif args.command == "init-db":
        logger.info("Initializing database...")
        create_tables()
        logger.info(f"Database initialized: {settings.database_url}")
        return 0
    
    elif args.command == "iot-test":
        from detectify.utils.iot import iot_manager
        logger.info(f"Testing IoT connection to {settings.esp32_ip}...")
        success = iot_manager.trigger_alert(class_name="test", confidence=1.0)
        if success:
            logger.info("✅ IoT Test Successful! ESP32 reached.")
        else:
            logger.error("❌ IoT Test Failed. Check IP and ESP32 status.")
        return 0 if success else 1
    
    elif args.command == "train":
        from detectify.train import run_training
        
        try:
            run_training(args.config)
        except NotImplementedError as e:
            logger.error(str(e))
            return 1
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
