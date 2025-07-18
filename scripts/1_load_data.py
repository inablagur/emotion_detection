# Import packages
from datasets import load_dataset
import pandas as pd
import os

# 1. Load the built-in “emotion” dataset
ds = load_dataset("emotion")
label_map = ds["train"].features["label"].names  # Get the label names

# 2. Ensure the data directory exists inside the main repo
os.makedirs("data", exist_ok=True)

# 3. For each split, convert to pandas df and save into a unique .csv file
splits = ["train", "validation", "test"]
for split in splits:
    df = pd.DataFrame(ds[split])
    # Map numeric labels to string names
    df["emotion"] = df["label"].apply(lambda i: label_map[i])
    # Keep only the columns needed and save to a dedicated .csv
    df[["text", "emotion"]].to_csv(f"data/{split}.csv", index=False)
    print(f"Saved data/{split}.csv  ({len(df)} rows)")
