# --------------------------------------------------------------------------------------------------------------------------------------------
# Import packages
import pandas as pd
import re
import contractions
import argparse
import pandas as pd
# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Clean Function ------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------ Modules -----------------------------------------------------------------
# 1. fix_informal_contractions
# Convert variants like im, i m, Im, Dont, don t (which exist in this dataset) to standardized forms (I'm, don't, etc.)

def fix_informal_contractions(text: str) -> str:
    """
    Normalize sloppy/missing-apostrophe contractions into their canonical forms
    for reliable downstream expansion. (e.g. im, i m, Im, I m → I'm etc.)

    Relevance:
      - Cleans noisy user text before other preprocessors.
      - ✔️ Shallow models: collapses 'dont' + "don't" into one token.
      - ✔️ Transformers: matches pretraining conventions for better embeddings.

    Args:
        text (str): Raw input string possibly containing informal contractions.

    Returns:
        str: Text with informal contractions normalized to canonical apostrophized forms.
    """
    # Mapping of regex → replacement
    pre_map = {
        # “I’m” variants
        r"\b(i\s?m|im|Im|I\s?m)\b": "I'm",
        # “don't” variants
        r"\b(dont|don\s?t|Dont|Don\s?t)\b": "don't",
        # “didn't”
        r"\b(didnt|did\s?nt|Didnt|Did\s?nt)\b": "didn't",
        # “I've”
        r"\b(ive|i\s?ve|Ive|I\s?ve)\b": "I've",
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

    # Apply each pattern in turn
    for pattern, replacement in pre_map.items():
        text = re.sub(pattern, replacement, text)
    return text

# --------------------------------------------------------------------------------------------------------------------------------------------
# 2. expand_contractions
# Use contractions.fix() to turn I'm → I am, don't → do not, etc.

def expand_contractions(text: str) -> str:
    """        
    Expand contractions into full-word forms for consistent tokenization (e.g. "don't", "Don't" → "do not" etc.)

    Relevance:
      - ✔️ Shallow models: unifies tokens to reduce feature sparsity.
      - ✔️ Transformer models: matches pretraining conventions

    Args:
        text (str): Input string with canonical apostrophized contractions.

    Returns:
        str: Text with all contractions expanded to their full forms.

    """
    return contractions.fix(text)


# --------------------------------------------------------------------------------------------------------------------------------------------
# 3. lowercase_text
# Normalize all characters to lowercase

def lowercase_text(text: str) -> str:
    """
    Convert all characters in the input string to lowercase for uniform casing (e.g. "Hello World!" → "hello world!" etc.)
    * Preserves non-letter characters (punctuation, numbers, emojis).

    Relevance:
      - ✔️ Shallow models: TF–IDF and count-vector features are case-sensitive by default so it reduces sparsity.
      - ✔️ Transformer models: although subword tokenizers are case-aware,
        lowercasing can improve consistency when using uncased variants (e.g. `bert-base-uncased`).

    Args:
        text (str): Input string with mixed casing.

    Returns:
        str: The same string, fully lowercased.
    """
    return text.lower()


# --------------------------------------------------------------------------------------------------------------------------------------------
# 4. normalize_whitespace
# Collapse runs of whitespace (\s+) to a single space and trim leading/trailing spaces.

def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into single spaces and strip edges (e.g. "  leading\nand trailing\t  " → "leading and trailing").

    Relevance:        
      - ✔️ Shallow models: collapsing irregular whitespace avoids empty tokens
        and reduces feature sparsity.
      - ✔️ Transformer models: ensures clean input for BERT tokenizers,
        matching pretraining assumptions and preserving sequence length for real content.        

    Args:
        text (str): Input string potentially containing irregular whitespace.

    Returns:
        str: Text with all runs of whitespace replaced by a single space,
             and no leading/trailing spaces.
    """
    return re.sub(r"\s+", " ", text).strip()      # \s+ matches any run of whitespace (spaces, tabs, newlines)

# --------------------------------------------------------------------------------------------------------------------------------------------
# 5. normalize_punctuation
# Replace repeated punctuation sequences (!!!, ..., ???) with a single character (!, ., ?).

def normalize_punctuation(text: str) -> str:
    """
    Collapse repeated punctuation and ensure spacing after punctuation marks (e.g. "Wait,what?No way..." → "Wait, what? No way." etc.)

    Relevance:
      - ✔️ Shallow models: avoids multiple tokens for “!”/“?” sequences and
        ensures punctuation is separate from words in TF–IDF vectors.
      - ✔️ Transformer models: aligns input with pretraining text conventions.

    Args:
        text (str): Input string with arbitrary punctuation.

    Returns:
        str: Text with repeated punctuation collapsed and a space after each mark.
    """
    # 1. Collapse sequences of the same punctuation to a single instance
    text = re.sub(r'([.!?;,]){2,}', r'\1', text)
    # 2. Ensure there is a space after punctuation if missing
    text = re.sub(r'([.!?;,])([^\s])', r'\1 \2', text)
    return text


# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- General DataFrame Functions --------------------------------------------------------

# 7. drop_invalid_rows
# Remove rows (from the entire df) where df[text] is null, empty, or only whitespace.

def drop_invalid_rows(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Remove rows where the text column is null, empty, or only whitespace.

    Relevance:
      - ✔️ Shallow models: avoids zero-vector documents that skew TF–IDF
      - ✔️ Transformer models: prevents tokenizers from receiving empty inputs,

    Args:
        df (pd.DataFrame): DataFrame containing a text column.
        text_col (str): Name of the column to validate and drop if invalid.

    Returns:
        pd.DataFrame: A new DataFrame with invalid rows removed and index reset.
    """
    # Convert to string, strip whitespace, then keep only non-empty entries
    valid_mask = df[text_col].notna() & df[text_col].astype(str).str.strip().astype(bool)
    return df.loc[valid_mask].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------------------------------
# 8. drop_by_length
# Remove rows whose text length is less than `min_len` or greater than `max_len`.

def drop_by_length(df: pd.DataFrame, text_col: str = 'text', min_len: int = 15, max_len: int = 300) -> pd.DataFrame:
    """
    Remove rows where the text length is outside [min_len, max_len].

    Relevance:
      - ✔️ Shallow models: filters out too-short noise and extreme outliers.
      - ✔️ Transformer models: filters out too-short noise and extreme outliers.

    Args:
        df (pd.DataFrame): Input DataFrame containing a text column.
        text_col (str): Name of the column to measure. Defaults to 'text'.
        min_len (int): Minimum valid character count. Defaults to 15 (as deduced from "notebooks\01_data_exploration.ipynb").
        max_len (int): Maximum valid character count. Defaults to 300 (as deduced from "notebooks\01_data_exploration.ipynb").

    Returns:
        pd.DataFrame: A new DataFrame with out-of-range rows removed and index reset.
    """
    lengths = df[text_col].astype(str).str.len()
    mask = lengths.between(min_len, max_len)

    return df.loc[mask].reset_index(drop=True)





