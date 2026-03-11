#!/bin/bash
# Export TF model to ONNX

if [ -z "$1" ]; then
    echo "Usage: $0 <saved_model_path> [output_path]"
    exit 1
fi

SAVED_MODEL=$1
OUTPUT=${2:-"model.onnx"}

echo "Converting $SAVED_MODEL to $OUTPUT..."
python -m tf2onnx.convert --saved-model "$SAVED_MODEL" --output "$OUTPUT" --opset 13

echo "Export complete!"
