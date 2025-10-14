"""
1_load_data.py - Load and prepare emotion detection dataset.

Loads the built-in "emotion" dataset from Hugging Face datasets,
converts numeric labels to string names, and saves each split as CSV.

Usage:
    python scripts/1_load_data.py --help
"""

# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------- Constants -----------------------------------------------------------------
OUTPUT_DIR = "data/raw"  # Directory to save the CSV files
SPLITS = ["train", "validation", "test"]  # Available dataset splits

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Load Function ------------------------------------------------------------
def load_emotion_dataset(output_dir: str = OUTPUT_DIR) -> None:
    """
    Load the emotion dataset and save each split as a CSV file.
    
    Args:
        output_dir (str): Directory to save the CSV files.
    """
    
    print("🔄 Loading emotion dataset...")
    
    # Load the built-in "emotion" dataset
    ds = load_dataset("emotion")
    splits = list(ds.keys())    # Get the splits
    label_map = ds["train"].features["label"].names  # Get the label names
    
    print(f"📊 Dataset loaded successfully!")
    print(f"📊 Available splits: {list(ds.keys())}")
    print(f"📊 Label mapping: {label_map}")
    
    # Ensure the data directory exists
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"📁 Creating directory: {output_path.resolve()}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory created successfully!")
    else:
        print(f"📁 Using existing directory: {output_path.resolve()}")
    
    # Collect all unique labels from all splits to ensure completeness
    all_labels = set()
    
    # Process each split
    print(f"\n📂 Saving splits to {output_path.resolve()}...")
    
    for split in splits:
        df = pd.DataFrame(ds[split])
        
        # Map numeric labels to string names
        df["emotion"] = df["label"].apply(lambda i: label_map[i])
        
        # Collect unique labels from this split
        all_labels.update(df["label"].unique())
        
        # Keep only the columns needed and save to a dedicated .csv
        output_file = output_path / f"{split}.csv"
        df[["text", "emotion", "label"]].to_csv(output_file, index=False)
        
        print(f"✅ Saved {output_file} ({len(df):,} rows)")

    # Create label2id mapping from all unique labels across all splits
    print(f"\n📋 Creating label2id mapping...")
    label2id = {label_map[i]: int(i) for i in sorted(all_labels)}  # Convert numpy int64 to Python int for JSON serialization
    
    print(f"📊 Found {len(label2id)} unique emotions across all splits:")
    for emotion, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"   {emotion}: {idx}")
    
    # Save label2id mapping to data/labels/label2id.json
    labels_dir = Path(output_dir).parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    label2id_file = labels_dir / "label2id.json"
    with open(label2id_file, 'w') as f:
        json.dump(label2id, f, indent=2)
    
    print(f"✅ Saved label2id mapping to {label2id_file}")
    
    print(f"\n🎉 Dataset loading completed successfully!")

# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------- Create Argument Parser ----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Load emotion detection dataset and save as CSV files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help=f"Directory to save CSV files (default: {OUTPUT_DIR})",
        default=OUTPUT_DIR
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    load_emotion_dataset(output_dir=args.output_dir)
