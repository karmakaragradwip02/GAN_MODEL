#!/bin/bash

# Clear pip cache
echo "Clearing pip cache..."
rm -rf ~/.cache/pip

# Clear TensorFlow cache
echo "Clearing TensorFlow cache..."
rm -rf ~/.keras
rm -rf ~/.cache/tensorflow

# Clear temporary files
echo "Clearing TensorFlow temporary files..."
rm -rf /tmp/tfhub_modules

# Remove Python bytecode
echo "Removing Python bytecode files..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.py[co]" -delete

echo "Cache clearing complete."
