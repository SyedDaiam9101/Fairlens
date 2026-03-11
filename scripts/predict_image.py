"""Predict Single Image CLI."""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectify.inference import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to image file")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()

    results = run_inference(source=args.path, output=args.output)
    print(f"Detections: {results.get('count', 0)}")

if __name__ == "__main__":
    main()
