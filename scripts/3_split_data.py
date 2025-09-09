"""
3_make_split.py - Create canonical data splits for emotion detection.

Creates a single source of truth for train/validation/test splits after data cleaning.
Generates data/splits/split_v1.csv with columns: id,split (values: train/val/test).
Splits are stratified by emotion label with fixed seed for reproducibility.

Usage:
    python scripts/3_make_split.py --help
"""

# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------- Constants -----------------------------------------------------------------
RANDOM_STATE = 42  # Fixed seed for reproducibility
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15   # 15% for validation  
TEST_RATIO = 0.15  # 15% for testing

# Verify ratios sum to 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Split ratios must sum to 1.0"

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Split Function ------------------------------------------------------------
def create_canonical_split(
    train_clean_path: str,
    val_clean_path: str, 
    test_clean_path: str,
    output_path: str,
    version: str = "v1"
) -> None:
    """
    Create canonical stratified split from cleaned data files.
    
    Args:
        train_clean_path (str): Path to cleaned training data
        val_clean_path (str): Path to cleaned validation data  
        test_clean_path (str): Path to cleaned test data
        output_path (str): Path to save split CSV
        version (str): Split version identifier (e.g., "v1", "v2")
    """
    
    print(f"🔄 Creating canonical split {version}...")
    
    # Load all cleaned data
    print("📂 Loading cleaned data files...")
    train_df = pd.read_csv(train_clean_path)
    val_df = pd.read_csv(val_clean_path)
    test_df = pd.read_csv(test_clean_path)
    
    # Add source information and create unique IDs
    train_df['source'] = 'train'
    val_df['source'] = 'val'
    test_df['source'] = 'test'
    
    # Create unique IDs based on source and index
    train_df['id'] = train_df.index.astype(str).str.zfill(6)
    val_df['id'] = (val_df.index + len(train_df)).astype(str).str.zfill(6)
    test_df['id'] = (test_df.index + len(train_df) + len(val_df)).astype(str).str.zfill(6)
    
    # Combine all data
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"📊 Combined data shape: {combined_df.shape}")
    print(f"📊 Emotion distribution:")
    print(combined_df['emotion'].value_counts().sort_index())
    
    # Create stratified split
    print("🎯 Creating stratified split...")
    
    # First split: separate train from temp (val + test)
    X = combined_df[['id', 'text', 'emotion']].copy()
    y = combined_df['emotion']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: separate val from test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # Proportion of temp that should be val
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    # Create split assignments
    split_assignments = []
    
    # Add training assignments
    for idx in X_train.index:
        split_assignments.append({
            'id': combined_df.loc[idx, 'id'],
            'split': 'train'
        })
    
    # Add validation assignments  
    for idx in X_val.index:
        split_assignments.append({
            'id': combined_df.loc[idx, 'id'],
            'split': 'val'
        })
    
    # Add test assignments
    for idx in X_test.index:
        split_assignments.append({
            'id': combined_df.loc[idx, 'id'],
            'split': 'test'
        })
    
    # Create final split DataFrame
    split_df = pd.DataFrame(split_assignments)
    split_df = split_df.sort_values('id').reset_index(drop=True)
    
    # Verify split proportions
    split_counts = split_df['split'].value_counts()
    total_samples = len(split_df)
    
    print(f"\n📈 Split distribution:")
    print(f"  Train: {split_counts['train']:,} ({split_counts['train']/total_samples:.1%})")
    print(f"  Val:   {split_counts['val']:,} ({split_counts['val']/total_samples:.1%})")
    print(f"  Test:  {split_counts['test']:,} ({split_counts['test']/total_samples:.1%})")
    
    # Verify stratification
    print(f"\n🎯 Stratification verification:")
    for split_name in ['train', 'val', 'test']:
        split_mask = split_df['split'] == split_name
        split_ids = split_df[split_mask]['id'].values
        split_emotions = combined_df[combined_df['id'].isin(split_ids)]['emotion']
        print(f"  {split_name.capitalize()} emotion distribution:")
        emotion_counts = split_emotions.value_counts().sort_index()
        for emotion, count in emotion_counts.items():
            pct = count / len(split_emotions) * 100
            print(f"    {emotion}: {count:,} ({pct:.1f}%)")
    
    # Save split file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Canonical split saved to: {output_path.resolve()}")
    print(f"📁 Split file contains {len(split_df):,} samples")
    
    return split_df

# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------- Create Argument Parser ----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Create canonical stratified data splits for emotion detection"
    )
    
    parser.add_argument(
        "--train-clean", "-t",
        # required=True,
        help="Path to cleaned training data CSV",
        default="data/clean/train_clean.csv"
    )
    
    parser.add_argument(
        "--val-clean", "-v", 
        # required=True,
        help="Path to cleaned validation data CSV",
        default="data/clean/validation_clean.csv"
    )
    
    parser.add_argument(
        "--test-clean", "-e",
        # required=True, 
        help="Path to cleaned test data CSV",
        default="data/clean/test_clean.csv"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save split CSV (default: data/splits/split_v1.csv)",
        default="data/splits/split_v1.csv"
    )
    
    parser.add_argument(
        "--version",
        help="Split version identifier (default: v1)",
        default="v1"
    )
    
    args = parser.parse_args()
    
    # Create canonical split
    split_df = create_canonical_split(
        train_clean_path=args.train_clean,
        val_clean_path=args.val_clean,
        test_clean_path=args.test_clean,
        output_path=args.output,
        version=args.version
    )
    
    print(f"\n🎉 Canonical split {args.version} created successfully!")



# TODO: Add, similar to 1_load_data.py, a check to see if the data/clean directory exists, and if not, create it and inform the user about it
# TODO: Make sure the split it valid. I think the data is already split into train, val, test, so we just need to make sure the split is valid, I'm afraid this split makes changes in the original split.