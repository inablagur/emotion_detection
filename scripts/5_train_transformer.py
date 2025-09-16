# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig
)
import uuid
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------- Constants ---------------------------------------------------------------
DEFAULT_TRAIN_DATA = "data/clean/train_clean.csv"
DEFAULT_VAL_DATA = "data/clean/validation_clean.csv"
DEFAULT_TEST_DATA = "data/clean/test_clean.csv"
DEFAULT_LABELS_FILE = "data/labels/label2id.json"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 5

def fail_fast_checks(train_data: str, val_data: str, test_data: str, labels_file: str) -> Tuple[bool, str]:
    """
    Perform fail-fast checks for required data files.
    
    Args:
        train_data: Path to training data file
        val_data: Path to validation data file
        test_data: Path to test data file
        labels_file: Path to labels file
    
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    required_files = [train_data, val_data, test_data, labels_file]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        error_msg = f"Missing required files: {', '.join(missing_files)}\n"
        error_msg += "Please run the data cleaning pipeline first:\n"
        error_msg += "  python scripts/2_clean_data.py"
        return False, error_msg
    
    # Check if files are not empty
    for file_path in required_files:
        if os.path.getsize(file_path) == 0:
            error_msg = f"File {file_path} is empty. Please run data cleaning pipeline first."
            return False, error_msg
    
    return True, ""

def load_data(train_data: str, val_data: str, test_data: str, labels_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load cleaned data and labels.
    
    Args:
        train_data: Path to training data file
        val_data: Path to validation data file
        test_data: Path to test data file
        labels_file: Path to labels file
    
    Returns:
        Tuple of (train_df, val_df, test_df, label2id)
    """
    print("Loading data...")
    
    # Load data
    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(val_data)
    test_df = pd.read_csv(test_data)
    
    # Load labels
    with open(labels_file, "r") as f:
        label2id = json.load(f)
    
    print(f"Loaded {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
    print(f"Labels: {label2id}")
    
    return train_df, val_df, test_df, label2id

class EmotionDataset(torch.utils.data.Dataset):
    """Dataset class for emotion classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Use basic tokenizer API for old versions
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length - 2:  # Account for [CLS] and [SEP]
            tokens = tokens[:self.max_length - 2]
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(0)  # PAD token
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FrozenEmotionClassifier(nn.Module):
    """
    Emotion classifier with frozen transformer encoder.
    Only the classification head learns during training.
    """
    
    def __init__(self, model_name: str, num_labels: int, hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer encoder
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Classification head - start with linear head
        # If needed, try MLP (Linear→GELU→Dropout→Linear)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Add config for HuggingFace compatibility
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = hidden_dropout_prob
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different output formats for old transformers versions
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler, use the [CLS] token (first token)
            pooled_output = outputs[0][:, 0]  # outputs[0] is the last_hidden_state
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model in HuggingFace format."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model architecture info
        with open(os.path.join(save_directory, 'model_info.json'), 'w') as f:
            json.dump({
                'model_type': 'FrozenEmotionClassifier',
                'frozen_encoder': True,
                'num_labels': self.config.num_labels
            }, f, indent=2)

def train_frozen_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      label2id: Dict[str, int], args) -> Tuple[Any, Dict[str, Any]]:
    """
    Train frozen model (encoder frozen, only head learns).
    
    Args:
        model_name: Name of the transformer model
        train_df: Training dataframe
        val_df: Validation dataframe  
        label2id: Label to ID mapping
        args: Training arguments
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    print(f"Training frozen {model_name} model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FrozenEmotionClassifier(model_name, len(label2id))
    model.to(device)
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        [label2id[label] for label in train_df['emotion']],
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        [label2id[label] for label in val_df['emotion']],
        tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop
    model.train()
    best_val_accuracy = 0
    patience_counter = 0
    early_stopping_patience = 3
    
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].item()
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            
            final_predictions.extend(predictions.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(final_labels, final_predictions)
    micro_f1 = f1_score(final_labels, final_predictions, average='micro')
    macro_f1 = f1_score(final_labels, final_predictions, average='macro')
    weighted_f1 = f1_score(final_labels, final_predictions, average='weighted')
    
    metrics = {
        'model': model_name,
        'mode': args.mode,
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'loss': avg_val_loss,
        'epochs_trained': epoch + 1,
        'classification_report': classification_report(
            final_labels, final_predictions, 
            target_names=list(label2id.keys()),
            output_dict=True
        )
    }
    
    return model, metrics, tokenizer

def save_results(model, tokenizer, metrics: Dict[str, Any], model_name: str, mode: str, 
                val_df: pd.DataFrame, label2id: Dict[str, int]):
    """Save metrics, plots, and checkpoints."""
    
    # Create run ID for file naming
    run_id = f"{model_name.replace('/', '_')}_{mode}"
    
    # Save metrics JSON
    metrics_file = f"reports/metrics/transformers/{run_id}_val.json"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Check if this is the best model so far
    winner_file = "reports/metrics/transformers/transformer_winner.json"
    is_best = False
    
    if os.path.exists(winner_file):
        with open(winner_file, 'r') as f:
            winner_metrics = json.load(f)
        # Use micro-F1 as primary metric for imbalanced datasets
        if metrics['micro_f1'] > winner_metrics.get('micro_f1', 0):
            is_best = True
    else:
        is_best = True
    
    if is_best:
        # Update winner metrics
        with open(winner_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"New best model! Updated {winner_file}")
        
        # Save winner checkpoint
        winner_dir = "models/transformers/transformer_winner"
        os.makedirs(winner_dir, exist_ok=True)
        
        # Save model and tokenizer in HuggingFace format
        model.save_pretrained(winner_dir)
        tokenizer.save_pretrained(winner_dir)
        print(f"Saved winner checkpoint to {winner_dir}")
    
    # Generate predictions for confusion matrix
    device = next(model.parameters()).device
    model.eval()
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        [label2id[label] for label in val_df['emotion']],
        tokenizer
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix plot
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label2id.keys()),
                yticklabels=list(label2id.keys()))
    plt.title(f'Confusion Matrix - {model_name} ({mode})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    plot_file = f"reports/plots/transformers/{run_id}_cm_val.png"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {plot_file}")

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Function ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train transformer models for emotion classification')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       choices=['distilbert-base-uncased', 'bert-base-uncased'],
                       help='Transformer model to use')
    parser.add_argument('--mode', type=str, default='frozen',
                       choices=['frozen', 'finetune', 'peft'],
                       help='Training mode')
    parser.add_argument('--train_data', type=str, default=DEFAULT_TRAIN_DATA,
                       help='Path to training data file')
    parser.add_argument('--val_data', type=str, default=DEFAULT_VAL_DATA,
                       help='Path to validation data file')
    parser.add_argument('--test_data', type=str, default=DEFAULT_TEST_DATA,
                       help='Path to test data file')
    parser.add_argument('--labels_file', type=str, default=DEFAULT_LABELS_FILE,
                       help='Path to labels file')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("Warning: No GPU detected. Training will be slow on CPU.")
    
    # Fail-fast checks
    print("Performing fail-fast checks...")
    success, error_msg = fail_fast_checks(args.train_data, args.val_data, args.test_data, args.labels_file)
    if not success:
        print(f"❌ {error_msg}")
        sys.exit(1)
    print("✅ All required files found")
    
    # Load data
    try:
        train_df, val_df, test_df, label2id = load_data(args.train_data, args.val_data, args.test_data, args.labels_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Train model based on mode
    if args.mode == 'frozen':
        try:
            model, metrics, tokenizer = train_frozen_model(args.model, train_df, val_df, label2id, args)
            save_results(model, tokenizer, metrics, args.model, args.mode, val_df, label2id)
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"❌ Mode '{args.mode}' not implemented yet. Only 'frozen' mode is available.")
        sys.exit(1)

if __name__ == "__main__":
    main()
