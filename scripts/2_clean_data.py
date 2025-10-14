# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import contractions
import pandas as pd
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------- Constatnts ---------------------------------------------------------------
MIN_NUM_CHARACHTERS = 15     # Minimum number of characters in a text row to be considered valid. Decided based on the observations in "notebooks/01_data_exploration.ipynb".
MAX_NUM_CHARACHTERS = 300    # Maximum number of characters in a text row to be considered valid. Decided based on the observations in "notebooks/01_data_exploration.ipynb".
TEXT_COLUMN_NAME = 'text'    # Default name of the text column in the DataFrame. This is the column that will be cleaned.
INPUT_DIR = "data/raw"       # Default directory to read the CSV files from.
OUTPUT_DIR = "data/clean"    # Default directory to save the cleaned CSV files.

# Mapping of regex → replacement - Relevant to the specific dataset, based on the observations in "notebooks/01_data_exploration.ipynb".
INFORMAL_CONTRACTION_MAP = {
        # “I’m” variants
        r"\b(i\s?m|im|Im|I\s?m)\b": "I'm",
        # “don't” variants
        r"\b(dont|don\s?t|Dont|Don\s?t)\b": "don't",
        # “didn't”
        r"\b(didnt|did\s?nt|Didnt|Did\s?nt)\b": "didn't",
        # “I've”
        r"\b(ive|i\s?ve|Ive|I\s?ve)\b": "I've",
        # “you've”
        r"\b(youve|you\s?ve|Youve|Y\s?ve)\b": "You've",
        # “wasn't”
        r"\b(wasnt|was\s?nt|Wasnt|Was\s?nt)\b": "wasn't",
        # “doesn't”
        r"\b(doesnt|does\s?nt|Doesnt|Does\s?nt)\b": "doesn't",
        # “shouldn't”
        r"\b(shouldnt|should\s?nt|Shouldnt|Should\s?nt)\b": "shouldn't",
        # “can't”
        r"\b(cant|can\s?t|Cant|Can\s?t)\b": "can't",
        # “won't”
        r"\b(wont|won\s?t|Wont|Won\s?t)\b": "won't",
        # “couldn't”
        r"\b(couldnt|could\s?nt|Couldnt|Could\s?nt)\b": "couldn't",
        # “wouldn't”
        r"\b(wouldnt|would\s?nt|Wouldnt|Would\s?nt)\b": "wouldn't",
    }

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Clean Function ------------------------------------------------------------
class CleanPipeline:
    """
    A pipeline for text cleaning. Call individual steps or the full batch/string methods.
    """

    def __init__(self, informal_map: dict = INFORMAL_CONTRACTION_MAP):
        self._informal_map = informal_map  # Mapping of informal contractions to their normalized forms.
        
    def fix_informal_contractions(self, text: str) -> str:        
        """
        Step 1.a: Normalize sloppy/missing-apostrophe contractions (e.g. im, i m, Im, I m → I'm etc.)

        Relevance:
        - ✔️ Shallow models: reduced sparsity.
        - ✔️ Transformers: matches pretraining conventions.

        Args:
            text (str): Raw input string possibly containing informal contractions.

        Returns:
            str: Text with informal contractions normalized to canonical apostrophized forms.
        """
        for pattern, replacement in self._informal_map.items():
            text = re.sub(pattern, replacement, text)
        return text

    def expand_contractions(self, text: str) -> str:
        """Step 1.b: Expand apostrophized contractions to full forms (e.g. "don't", "Don't" → "do not" etc.)

        Relevance:
        - ✔️ Shallow models: reduce feature sparsity.
        - ✔️ Transformer models: matches pretraining conventions

        Args:
            text (str): Input string with canonical apostrophized contractions.

        Returns:
            str: Text with all contractions expanded to their full forms.
        """
        return contractions.fix(text)

    def lowercase_text(self, text: str) -> str:
        """Step 2: Lowercase the text (e.g. "Hello World!" → "hello world!" etc.)
        * Preserves non-letter characters (punctuation, numbers, emojis).

        Relevance:
        - ✔️ Shallow models: reduces sparsity.
        - ✔️ Transformer models: can improve consistency when using uncased variants (e.g. `bert-base-uncased`).

        Args:
            text (str): Input string with mixed casing.

        Returns:
            str: The same string, fully lowercased.        
        """
        return text.lower()

    def normalize_whitespace(self, text: str) -> str:
        """Step 3: Collapse whitespace runs and strip edges (e.g. "  leading\nand trailing\t  " → "leading and trailing").

        Relevance:        
        - ✔️ Shallow models: reduces feature sparsity.
        - ✔️ Transformer models: ensures clean input for BERT tokenizers,
            matching pretraining assumptions and preserving sequence length for real content.        

        Args:
            text (str): Input string potentially containing irregular whitespace.

        Returns:
            str: Text with all runs of whitespace replaced by a single space, and no leading/trailing spaces.        
        """
        return re.sub(r"\s+", " ", text).strip()    # \s+ matches any run of whitespace (spaces, tabs, newlines)


    def normalize_punctuation(self, text: str) -> str:
        """Step 4: Collapse repeated punctuation and ensure spacing after punctuation marks (e.g. "Wait,what?No way..." → "Wait, what? No way." etc.)

        Relevance:
        - ✔️ Shallow models: avoids multiple tokens for “!”/“?” sequences and
            ensures punctuation is separate from words in TF-IDF vectors.
        - ✔️ Transformer models: aligns input with pretraining text conventions.

        Args:
            text (str): Input string with arbitrary punctuation.

        Returns:
            str: Text with repeated punctuation collapsed and a space after each mark.
        """
        # 1. Collapse sequences of the same punctuation to a single instance
        text = re.sub(r'([.!?;,]){2,}', r'\1', text)
        
        # 2. Ensure there is a space after punctuation if missing
        return re.sub(r'([.!?;,])([^\s])', r'\1 \2', text)

    def drop_invalid_rows(self, df: pd.DataFrame, text_col: str = TEXT_COLUMN_NAME) -> pd.DataFrame:
        """Step 5: Drop rows where text is null / empty / whitespace only.

        Relevance:
        - ✔️ Shallow models: avoids zero-vector documents that skew TF-IDF
        - ✔️ Transformer models: prevents tokenizers from receiving empty inputs,

        Args:
            df (pd.DataFrame): DataFrame containing a text column.
            text_col (str): Name of the column to validate and drop if invalid.

        Returns:
            pd.DataFrame: A new DataFrame with invalid rows removed and index reset.
        """
        
        self.text_col = text_col
        # Convert to string, strip whitespace, then keep only non-empty entries
        valid_mask = df[self.text_col].notna() & df[self.text_col].astype(str).str.strip().astype(bool)
        return df.loc[valid_mask].reset_index(drop=True)

    def drop_by_length(self, df: pd.DataFrame, text_col: str = TEXT_COLUMN_NAME, min_len: int = MIN_NUM_CHARACHTERS, max_len: int = MAX_NUM_CHARACHTERS) -> pd.DataFrame:
        """Step 6: Remove rows where the text length is outside [min_len, max_len].

        Relevance:
        - ✔️ Shallow models: filters out too-short noise and extreme outliers.
        - ✔️ Transformer models: filters out too-short noise and extreme outliers.

        Args:
            df (pd.DataFrame): Input DataFrame containing a text column.
            text_col (str): Name of the column to measure. Defaults to 'text'.
            min_len (int): Minimum valid character count.
            max_len (int): Maximum valid character count. 

        Returns:
            pd.DataFrame: A new DataFrame with out-of-range rows removed and index reset.
        """
        
        self.min_len = min_len
        self.max_len = max_len
        self.text_col = text_col

        lengths = df[self.text_col].astype(str).str.len()
        mask = lengths.between(self.min_len, self.max_len)
        return df.loc[mask].reset_index(drop=True)

    # Main method to clean a DataFrame, relevant for batch processing and training.
    def clean_df(self, df: pd.DataFrame, text_col: str = TEXT_COLUMN_NAME, min_len: int = MIN_NUM_CHARACHTERS, max_len: int = MAX_NUM_CHARACHTERS) -> pd.DataFrame:
        """
        Apply steps 1-4 to every row, then steps 5-6 for pruning.
        """
        
        self.text_col = text_col
        df = df.copy()
        df[self.text_col] = df[self.text_col].fillna('').astype(str)

        for fn in (
            self.fix_informal_contractions,
            self.expand_contractions,
            self.lowercase_text,
            self.normalize_whitespace,
            self.normalize_punctuation
        ):
            df[self.text_col] = df[self.text_col].apply(fn)

        df = self.drop_invalid_rows(df, self.text_col)
        df = self.drop_by_length(df, self.text_col, min_len, max_len)
        return df

    # This method is for single-string cleaning. relevant for inference or interactive use.
    def clean_text(self, text: str, min_len: int = MIN_NUM_CHARACHTERS, max_len: int = MAX_NUM_CHARACHTERS) -> str:
        """
        Apply steps 1-4 to a single string, then validate length & non-emptiness.
        Raises ValueError on invalid input.
        """
        
        self.min_len = min_len
        self.max_len = max_len
        
        for fn in (
            self.fix_informal_contractions,
            self.expand_contractions,
            self.lowercase_text,
            self.normalize_whitespace,
            self.normalize_punctuation
        ):
            text = fn(text)

        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text is empty after cleaning.")
        if not self.min_len <= len(stripped) <= self.max_len:
            raise ValueError(f"Length {len(stripped)} outside range [{self.min_len},{self.max_len}]")
        return stripped


# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------- Create Argument Parser ----------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Clean text data via CleanPipeline. Processes all files in data/raw/ or a specific file."
    )

    parser.add_argument(
        "--input-csv", "-i",
        required=False,
        default=None,
        help="Path to input CSV file. If not provided, processes all CSV files in data/raw/"
    )

    parser.add_argument(
        "--output-csv", "-o",
        help="Path to write cleaned CSV. If omitted, saves to data/clean/ with original filename + '_clean' suffix.",
        default=None
    )

    args = parser.parse_args()

    # Instantiate pipeline
    pipe = CleanPipeline()

    # Determine which files to process
    if args.input_csv:
        # Single file mode
        input_files = [Path(args.input_csv)]
        print(f"📄 Processing single file: {args.input_csv}")
    else:
        # Batch mode - process all CSV files in data/raw/
        raw_dir = Path(INPUT_DIR)
        if not raw_dir.exists():
            print(f"❌ Error: {raw_dir} directory does not exist")
            print("   Please run 'python scripts/1_load_data.py' first")
            sys.exit(1)
        
        input_files = list(raw_dir.glob("*.csv"))
        if not input_files:
            print(f"❌ Error: No CSV files found in {raw_dir}")
            sys.exit(1)
        
        print(f"📂 Processing all files in {raw_dir}:")
        for f in input_files:
            print(f"   - {f.name}")
    
    # Process each file
    for in_path in input_files:
        print(f"\n🔄 Cleaning {in_path.name}...")
        
        # Read the CSV
        df = pd.read_csv(in_path)
        
        # Clean it
        cleaned = pipe.clean_df(df)
        
        # Determine output path
        if args.output_csv and len(input_files) == 1:
            # Single file with specified output
            out_path = Path(args.output_csv)
        else:
            # Save to data/clean/ with _clean suffix
            clean_dir = Path(OUTPUT_DIR)
            clean_dir.mkdir(parents=True, exist_ok=True)
            out_path = clean_dir / f"{in_path.stem}_clean.csv"
        
        # Save to CSV
        cleaned.to_csv(out_path, index=False)
        print(f"✅ Cleaned CSV written to: {out_path.resolve()}")
    
    print(f"\n🎉 All files cleaned successfully!")