#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./compile_and_run.sh <file.cu>"
    exit 1
fi

FILE=$1
BASENAME="${FILE%.cu}"

nvcc "$FILE" -o "$BASENAME" && ./"$BASENAME" && rm "$BASENAME"
