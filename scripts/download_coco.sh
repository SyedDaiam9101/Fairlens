#!/bin/bash
# Download COCO 2017 val set (for testing)

mkdir -p data/val2017
mkdir -p data/annotations

echo "Downloading COCO annotations..."
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o data/annotations.zip
unzip data/annotations.zip -d data/
rm data/annotations.zip

echo "Downloading COCO val2017 images (1GB)..."
# curl -L http://images.cocodataset.org/zips/val2017.zip -o data/val2017.zip
# unzip data/val2017.zip -d data/
# rm data/val2017.zip

echo "Done!"
