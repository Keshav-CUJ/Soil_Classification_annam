#!/bin/bash

# ============================
# Script to Download Kaggle Dataset
# ============================

# Required: Kaggle dataset identifier (e.g., 'annam-ai/soilclassification')
KAGGLE_DATASET="annam-ai/soil-classification"
# Target download directory
TARGET_DIR="./data"

echo "📥 Starting download from Kaggle: $KAGGLE_DATASET"

# Ensure Kaggle API credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle API credentials not found! Please place kaggle.json in ~/.kaggle/"
    exit 1
fi

# Ensure the proper file permissions
chmod 600 ~/.kaggle/kaggle.json

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download and unzip the dataset
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

if [ $? -eq 0 ]; then
    echo "✅ Download complete. Files saved to $TARGET_DIR"
else
    echo "❌ Download failed. Please check the dataset name or your internet connection."
    exit 1
fi
