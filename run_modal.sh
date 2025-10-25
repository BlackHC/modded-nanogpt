#!/bin/bash

# Check if user wants to list logs
if [ "$1" = "logs" ]; then
    echo "Listing logs from Modal volume..."
    python modal_train.py logs
    exit 0
fi

# Run Modal training with 8 H100s
echo "Starting Modal-based multi-GPU training..."
python modal_train.py 