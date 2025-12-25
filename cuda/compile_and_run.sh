#!/bin/bash

# Script to compile, run, and clean up CUDA programs
# Usage: ./compile_and_run.sh <file.cu>
# Example: ./compile_and_run.sh 03-memory-basics.cu

if [ $# -eq 0 ]; then
    echo "Usage: ./compile_and_run.sh <file.cu>"
    echo "Example: ./compile_and_run.sh 03-memory-basics.cu"
    exit 1
fi

FILE=$1

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found"
    exit 1
fi

# Check if file has .cu extension
if [[ ! "$FILE" =~ \.cu$ ]]; then
    echo "Error: File must have .cu extension"
    exit 1
fi

# Extract basename without extension
BASENAME="${FILE%.cu}"

echo "Compiling $FILE..."
nvcc "$FILE" -o "$BASENAME"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo ""
echo "Running $BASENAME..."
echo "----------------------------------------"
./"$BASENAME"
EXIT_CODE=$?
echo "----------------------------------------"
echo ""

# Clean up
rm "$BASENAME"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Program completed successfully"
else
    echo "✗ Program exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
