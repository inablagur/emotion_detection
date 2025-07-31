# --------------------------------------------------------------------------------------------------------------------------------------------
# Import packages
import pandas as pd
import re
import contractions
import argparse

# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# Main Clean Function:



# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# 1. fix_informal_contractions(text: str) -> str
# Convert variants like im, i m, Im, Dont, don t (which exist in this dataset) → standardized forms (I'm, don't, etc.)

def fix_informal_contractions(text: str) -> str:
    """
    Normalize sloppy/missing-apostrophe contractions into their canonical forms
    for reliable downstream expansion.

    Specifically handles:
      - im, i m, Im, I m       → I'm
      - dont, don t, Dont, Don t → don't
      - (and similarly for: didnt, ive, wasnt, doesnt, shouldnt, cant, wont, couldnt, wouldn't)

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
        # We don’t set IGNORECASE because we explicitly list capitalized forms
        text = re.sub(pattern, replacement, text)
    return text

# --------------------------------------------------------------------------------------------------------------------------------------------
def expand_contractions(text: str) -> str:
    """
    Expand canonical apostrophized contractions into their full-word forms,
    so that downstream tokenization and modeling see consistent vocabulary.

    Specifically handles patterns like:
      - "don't", "Don't"   → "do not"
      - "I'm", "i'm"       → "I am"
      - "they're", "They’re" → "they are"
      - (and similarly for: you've, he'd, she'd, we'll, won't, etc.)

    We rely on the `contractions` package’s built-in mapping of ~150+ cases
    to cover nested and edge-case contractions, avoiding manual maintenance.

    Relevance:
      - **Shallow models** (e.g. TF–IDF + LogisticRegression/SVM) benefit from
        reduced vocabulary sparsity: "don't" and "do not" become the same tokens.
      - **Transformer models** (e.g. BERT, DistilBERT) align better with
        pretraining data (which often uses full forms), improving embedding quality.
        
        
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

