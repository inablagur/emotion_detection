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
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------- Constants -----------------------------------------------------------------
DATA_DIR = "data/original"  # Directory to save the CSV files
SPLITS = ["train", "validation", "test"]  # Available dataset splits

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Load Function ------------------------------------------------------------
def load_emotion_dataset(output_dir: str = DATA_DIR) -> None:
    """
    Load the emotion dataset and save each split as a CSV file.
    
    Args:
        output_dir (str): Directory to save the CSV files.
    """
    
    print("🔄 Loading emotion dataset...")
    
    # Load the built-in "emotion" dataset
    ds = load_dataset("emotion")
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
    
    # Process each split
    print(f"\n📂 Saving splits to {output_path.resolve()}...")
    
    for split in SPLITS:
        if split in ds:
            df = pd.DataFrame(ds[split])
            
            # Map numeric labels to string names
            df["emotion"] = df["label"].apply(lambda i: label_map[i])
            
            # Keep only the columns needed and save to a dedicated .csv
            output_file = output_path / f"{split}.csv"
            df[["text", "emotion"]].to_csv(output_file, index=False)
            
            print(f"✅ Saved {output_file} ({len(df):,} rows)")
        else:
            print(f"⚠️  Split '{split}' not found in dataset")
    
    print(f"\n🎉 Dataset loading completed successfully!")

# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------- Create Argument Parser ----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Load emotion detection dataset and save as CSV files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help=f"Directory to save CSV files (default: {DATA_DIR})",
        default=DATA_DIR
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    load_emotion_dataset(output_dir=args.output_dir)
