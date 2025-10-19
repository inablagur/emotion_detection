# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------- Import Packages --------------------------------------------------------------
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import traceback
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

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

# Obligatory CSV columns - these must always exist in results CSV files
# Column names are fixed and cannot be changed. Values can be null/NaN if not available.
OBLIGATORY_CSV_COLUMNS = [
    'experiment',           # Sequential experiment number
    'model',               # Model name (e.g., distilbert-base-uncased)
    'method',              # Training method (frozen, finetune, peft)
    'learning_rate',       # Learning rate hyperparameter
    'batch_size',          # Batch size hyperparameter
    'epochs',              # Number of epochs hyperparameter
    'loss_type',           # Loss function type (weighted, standard)
    'weight_decay',        # L2 regularization weight
    'max_length',          # Token sequence max length
    'val_micro_f1',        # Validation set micro-averaged F1 (primary metric)
    'val_macro_f1',        # Validation set macro-averaged F1
    'val_accuracy',        # Validation set accuracy
    'training_time',       # Total training time in seconds
    'status',              # Experiment status (success, failed)
    'is_winner',           # Boolean: True if this is the best experiment in this search
]

# Default hyperparameter grids for randomized search
# Currently 'frozen' and 'peft' methods are implemented
# When adding new methods (finetune), define their grids here with appropriate ranges
DEFAULT_HYPERPARAMETER_GRIDS = {
    'frozen': {
        'learning_rate': [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],  # Higher LR since only training head
        'batch_size': [8, 16, 32],   # Higher batch size isn't worth the memory usage. The current size is enough for this dataset size
        'epochs': [3, 4, 5, 6],
        'loss_type': ['weighted', 'standard'],
        'weight_decay': [0.0, 0.01, 0.05, 0.1]  # L2 regularization. Includes higher values to avoid potential overfitting (due to small number of trainable parameters)
    },
    'peft': {
        # Standard hyperparameters
        'learning_rate': [1e-4, 2e-4, 5e-4],  # *Lower than frozen - training more parameters
        'batch_size': [8, 16, 32],  # Higher batch size isn't worth the memory usage. The current size is enough for this dataset size
        'epochs': [3, 5, 7],  # May need more epochs than frozen
        'loss_type': ['weighted', 'standard'],
        'weight_decay': [0.0, 0.01],  # Include only lower values since it less prones to overfit 
        
        # LoRA-specific hyperparameters
        'lora_r': [4, 8, 16, 32],  # Rank: low (4-8) for efficiency, high (16-32) for expressiveness
        'lora_alpha': [8, 16, 32],  # Scaling factor (typically 1-2x the rank)
        'lora_dropout': [0.05, 0.1],  # Dropout for LoRA layers (regularization)
        
        # Note: lora_target_modules is FIXED to ['query', 'value'] (standard approach)
        # TODO: If results are not satisfactory, consider testing more aggressive target modules (and decide whether to implement them in the grid search or someplace else):
        #       ['query', 'key', 'value', 'dense'] for ~2x more trainable parameters at cost of slower training
    }
    # 'finetune': {...}  # To be added when implementing full fine-tuning
}

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
    print("\nLoading data...")
    
    # Load data
    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(val_data)
    test_df = pd.read_csv(test_data)
    
    # Load labels
    with open(labels_file, "r") as f:
        label2id = json.load(f)
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"Loaded {len(train_df)} ({np.round((len(train_df) / total_samples) * 100)}%) train, {len(val_df)} ({np.round((len(val_df) / total_samples) * 100)}%) validation, {len(test_df)} ({np.round((len(test_df) / total_samples) * 100)}%) test")
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
        
        # Use modern tokenizer API (works with transformers>=4.20.0)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FrozenEmotionClassifier(nn.Module):
    """
    Emotion classifier with frozen transformer encoder.
    Only the classification head learns during training.
    Handles imbalanced data
    """
    
    def __init__(self, model_name: str, num_labels: int, hidden_dropout_prob: float = 0.1, 
                 class_weights=None):
        """
        Initialize frozen emotion classifier.
        
        Args:
            model_name: HuggingFace model name (e.g., 'distilbert-base-uncased')
            num_labels: Number of emotion classes
            hidden_dropout_prob: Dropout for classification head (default 0.1 - community standard, proven effective across tasks)
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)     # This attribute is used for saving the model config (including the base model and the additions we add)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer encoder - SAVES COMPUTATION TIME
        print(f"🧊 Freezing transformer encoder parameters...")
        for param in self.transformer.parameters():
            param.requires_grad = False
        print(f"✅ Frozen {sum(1 for p in self.transformer.parameters())} parameters")
        
        # Classification head - start with linear head
        # TODO: Experiment with MLP head architecture (Linear→GELU→Dropout→Linear) for better performance
        # TODO: Optional: Make head architecture configurable via hyperparameter grid
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Loss function for imbalanced data (simplified - only weighted or standard)
        if class_weights is not None:
            self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            print("⚖️ Using weighted CrossEntropyLoss for imbalanced data")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            print("📈 Using standard CrossEntropyLoss")
        
        # Add config for HuggingFace compatibility
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = hidden_dropout_prob
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            labels: Ground truth labels [batch_size] (optional)
            
        Returns:
            Dict with 'loss' and 'logits'
        """
        # Get transformer outputs (FROZEN - no gradients computed for encoder)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different output formats for compatibility
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler, use the [CLS] token (first token)
            pooled_output = outputs.last_hidden_state[:, 0]  # Extract [CLS] token
        
        # Apply dropout and classification (TRAINABLE PART)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model in HuggingFace format - SAVES TIME during model loading."""
        print(f"💾 Saving model to {save_directory}...")
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
                'num_labels': self.config.num_labels,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
        print(f"✅ Model saved successfully\n")


class PEFTEmotionClassifier(nn.Module):
    """
    Emotion classifier using PEFT (LoRA) for parameter-efficient fine-tuning.
    Applies LoRA to attention layers while keeping most of the model frozen.
    Only LoRA adapters and classification head learn during training.
    Handles imbalanced data.
    """
    
    def __init__(self, model_name: str, num_labels: int, 
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 lora_target_modules: list = None,
                 hidden_dropout_prob: float = 0.1, 
                 class_weights=None):
        """
        Initialize PEFT emotion classifier with LoRA.
        
        Args:
            model_name: HuggingFace model name (e.g., 'distilbert-base-uncased')
            num_labels: Number of emotion classes
            lora_r: LoRA rank (controls adapter capacity)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability for LoRA layers
            lora_target_modules: Which modules to apply LoRA to (default: ['query', 'value'])
            hidden_dropout_prob: Dropout for classification head (default 0.1 - community standard, proven effective across tasks)
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        
        # Load base model and config
        self.config = AutoConfig.from_pretrained(model_name)    # This attribute is used for saving the model config (including the base model and the additions we add)
        base_model = AutoModel.from_pretrained(model_name)
        
        # Set default target modules if not provided
        if lora_target_modules is None:
            lora_target_modules = ['query', 'value']  # Standard approach
        
        # Configure LoRA
        print(f"🔧 Configuring LoRA with rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"🎯 Target modules: {lora_target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # We add custom classification head
            r=lora_r,  # Rank
            lora_alpha=lora_alpha,  # Scaling factor
            lora_dropout=lora_dropout,  # Dropout
            target_modules=lora_target_modules,  # Which attention matrices to adapt
            bias="none",  # Don't adapt biases (Adapting biases adds minimal capacity but more parameters)
            inference_mode=False  # Training mode
        )
        
        # Apply LoRA to the base model
        self.transformer = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        trainable_percent = 100 * trainable_params / total_params
        print(f"✅ LoRA applied: {trainable_params:,} trainable parameters ({trainable_percent:.3f}% of {total_params:,} total)")
        
        # Add classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Loss function for imbalanced data
        if class_weights is not None:
            self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            print("⚖️ Using weighted CrossEntropyLoss for imbalanced data")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            print("📈 Using standard CrossEntropyLoss")
        
        # Store LoRA config in model config for saving
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.lora_r = lora_r
        self.config.lora_alpha = lora_alpha
        self.config.lora_dropout = lora_dropout
        self.config.lora_target_modules = lora_target_modules
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)
        
        Returns:
            Dict with 'loss' and 'logits'
        """
        # Get transformer outputs with LoRA adapters
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different output formats for compatibility
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler, use the [CLS] token (first token)
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classification (TRAINABLE PART)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def save_pretrained(self, save_directory: str):
        """Save PEFT model in HuggingFace format."""
        print(f"💾 Saving PEFT model to {save_directory}...")
        os.makedirs(save_directory, exist_ok=True)
        
        # Save LoRA adapters using PEFT's built-in save method
        self.transformer.save_pretrained(save_directory)
        
        # Save classification head and config
        torch.save({
            'classifier': self.classifier.state_dict(),
            'dropout': self.dropout.state_dict()
        }, os.path.join(save_directory, 'classification_head.bin'))
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model architecture info
        with open(os.path.join(save_directory, 'model_info.json'), 'w') as f:
            json.dump({
                'model_type': 'PEFTEmotionClassifier',
                'peft_type': 'LoRA',
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'lora_dropout': self.config.lora_dropout,
                'lora_target_modules': self.config.lora_target_modules,
                'num_labels': self.config.num_labels,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
        print(f"✅ PEFT model saved successfully\n")


def compute_class_weights(train_labels: list, label2id: Dict[str, int]) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    ADDRESSES CLASS IMBALANCE - gives more weight to minority classes.
    """
    print("⚖️ Computing class weights for imbalanced data...")
    
    # Convert string labels to numeric
    numeric_labels = [label2id[label] for label in train_labels]
    
    # Compute weights
    classes = np.array(list(range(len(label2id))))
    weights = compute_class_weight('balanced', classes=classes, y=numeric_labels)
    
    # Convert to tensor
    class_weights = torch.FloatTensor(weights)
    
    # Print class distribution
    unique, counts = np.unique(numeric_labels, return_counts=True)
    id2label = {v: k for k, v in label2id.items()}
    
    print("📊 Class distribution and weights:")
    for class_id, count in zip(unique, counts):
        emotion = id2label[class_id]
        weight = weights[class_id]
        print(f"   {emotion}: {count} samples (weight: {weight:.3f})")
    print("\n")
    return class_weights

def train_frozen_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      label2id: Dict[str, int], args) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Train frozen model (encoder frozen, only head learns).
    OPTIMIZED for optional imbalanced data and faster training.
    
    Args:
        model_name: Name of the transformer model
        train_df: Training dataframe
        val_df: Validation dataframe  
        label2id: Label to ID mapping
        args: Training arguments
        
    Returns:
        Tuple of (model, metrics_dict, tokenizer)
    """
    print(f"🚀 Training frozen {model_name} model...")
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load tokenizer 
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Compute class weights
    # Handle imbalanced data based on loss_type argument
    if args.loss_type == 'weighted':
        class_weights = compute_class_weights(train_df['emotion'].tolist(), label2id)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
        print("📈 Using standard CrossEntropyLoss (no imbalance handling)")
    
    # Create model with weighted loss (simple and effective)
    model = FrozenEmotionClassifier(
        model_name, 
        len(label2id), 
        class_weights=class_weights
    )
    model.to(device)
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer - ONLY trains classification head (faster)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = num_trainable + num_frozen
    print(f"🎯 Model parameters: {total_params:,} total | {num_trainable:,} trainable | {num_frozen:,} frozen")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    print(f"🎯 Using learning rate: {args.learning_rate}")
    
    # Training loop with early stopping
    model.train()
    best_val_f1 = 0
    patience_counter = 0
    early_stopping_patience = 3
    # TODO: Add optional early_stopping and value (either to the grid search or as an individual argument)
    
    print(f"🏃‍♂️ Starting training for {args.epochs} epochs...\n")
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
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
        
        with torch.no_grad(): # Don't compute gradients (faster, saves memory)
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].item()
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())  # Convert to cpu + numpy array to allow scikit-learn evaluation
                val_labels.extend(labels.cpu().numpy()) # Convert to cpu + numpy array to allow scikit-learn evaluation
        
        # Save predictions for final metrics calculation
        last_val_predictions = val_predictions
        last_val_labels = val_labels
        
        # Calculate only metrics needed for early stopping and progress tracking
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_micro_f1 = f1_score(val_labels, val_predictions, average='micro')
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Micro-F1: {val_micro_f1:.4f} (Time: {epoch_time:.1f}s)')
        
        # Early stopping based on micro-F1
        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Calculate additional validation metrics once (using last epoch's predictions)
    print("\n📊 Calculating final validation metrics...")
    val_accuracy = accuracy_score(last_val_labels, last_val_predictions)
    val_micro_f1 = f1_score(last_val_labels, last_val_predictions, average='micro')
    val_macro_f1 = f1_score(last_val_labels, last_val_predictions, average='macro')
    val_weighted_f1 = f1_score(last_val_labels, last_val_predictions, average='weighted')
    
    
    total_time = time.time() - start_time
    training_time = time.time() - training_start
    
    metrics = {
        'model': model_name,
        'mode': args.mode,
        # Validation metrics (primary for model selection)
        'val_accuracy': val_accuracy,
        'val_micro_f1': val_micro_f1,  # PRIMARY metric for imbalanced data
        'val_macro_f1': val_macro_f1,
        'val_weighted_f1': val_weighted_f1,
        # Other info
        'loss': avg_val_loss,
        'epochs_trained': epoch + 1,
        'total_time_seconds': total_time,
        'training_time_seconds': training_time,
        'classification_report': classification_report(
            last_val_labels, last_val_predictions, 
            target_names=list(label2id.keys()),
            output_dict=True
        )
    }
    
    print(f"✅ Training completed in {total_time:.1f}s (training: {training_time:.1f}s)")
    print(f"📊 Validation Metrics - Accuracy: {val_accuracy:.3f}, Micro-F1: {val_micro_f1:.3f}, Macro-F1: {val_macro_f1:.3f}")
    
    return model, metrics, tokenizer


def train_peft_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                    label2id: Dict[str, int], args) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Train PEFT model (LoRA adapters + classification head learn).
    Uses parameter-efficient fine-tuning with LoRA adapters.
    Handles imbalanced data and provides detailed training progress.
    
    Args:
        model_name: Name of the transformer model
        train_df: Training dataframe
        val_df: Validation dataframe  
        label2id: Label to ID mapping
        args: Training arguments (includes LoRA hyperparameters)
        
    Returns:
        Tuple of (model, metrics_dict, tokenizer)
    """
    print(f"🚀 Training PEFT {model_name} model...")
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load tokenizer 
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Compute class weights
    # Handle imbalanced data based on loss_type argument
    if args.loss_type == 'weighted':
        class_weights = compute_class_weights(train_df['emotion'].tolist(), label2id)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
        print("📈 Using standard CrossEntropyLoss (no imbalance handling)")
    
    # Create PEFT model with LoRA configuration
    model = PEFTEmotionClassifier(
        model_name, 
        len(label2id),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        class_weights=class_weights
    )
    model.to(device)
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer - trains LoRA adapters AND classification head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = num_trainable + num_frozen
    print(f"🎯 Model parameters: {total_params:,} total | {num_trainable:,} trainable | {num_frozen:,} frozen")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"🎯 Using learning rate: {args.learning_rate}, weight decay: {args.weight_decay}")
    
    # Training loop with early stopping
    model.train()
    best_val_f1 = 0
    patience_counter = 0
    early_stopping_patience = 3
    # TODO: Add optional early_stopping and value (either to the grid search or as an individual argument)
    
    print(f"🏃‍♂️ Starting PEFT training for {args.epochs} epochs...\n")
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
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
        
        with torch.no_grad(): # Don't compute gradients (faster, saves memory)
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].item()
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())  # Convert to cpu + numpy array to allow scikit-learn evaluation
                val_labels.extend(labels.cpu().numpy()) # Convert to cpu + numpy array to allow scikit-learn evaluation
        
        # Save predictions for final metrics calculation
        last_val_predictions = val_predictions
        last_val_labels = val_labels
        
        # Calculate only metrics needed for early stopping and progress tracking
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_micro_f1 = f1_score(val_labels, val_predictions, average='micro')
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Micro-F1: {val_micro_f1:.4f} (Time: {epoch_time:.1f}s)')
        
        # Early stopping based on micro-F1
        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Calculate additional validation metrics once (using last epoch's predictions)
    print("\n📊 Calculating final validation metrics...")
    val_accuracy = accuracy_score(last_val_labels, last_val_predictions)
    val_micro_f1 = f1_score(last_val_labels, last_val_predictions, average='micro')
    val_macro_f1 = f1_score(last_val_labels, last_val_predictions, average='macro')
    val_weighted_f1 = f1_score(last_val_labels, last_val_predictions, average='weighted')
    
    total_time = time.time() - start_time
    training_time = time.time() - training_start
    
    metrics = {
        'model': model_name,
        'mode': args.mode,
        # Validation metrics (primary for model selection)
        'val_accuracy': val_accuracy,
        'val_micro_f1': val_micro_f1,  # PRIMARY metric for imbalanced data
        'val_macro_f1': val_macro_f1,
        'val_weighted_f1': val_weighted_f1,
        # Other info
        'loss': avg_val_loss,
        'epochs_trained': epoch + 1,
        'total_time_seconds': total_time,
        'training_time_seconds': training_time,
        'classification_report': classification_report(
            last_val_labels, last_val_predictions, 
            target_names=list(label2id.keys()),
            output_dict=True
        )
    }
    
    print(f"✅ PEFT training completed in {total_time:.1f}s (training: {training_time:.1f}s)")
    print(f"📊 Validation Metrics - Accuracy: {val_accuracy:.3f}, Micro-F1: {val_micro_f1:.3f}, Macro-F1: {val_macro_f1:.3f}")
    
    return model, metrics, tokenizer


# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Training Dispatcher & Randomized Search ------------------------------------------------
def train_model(method: str, model_name: str, train_df: pd.DataFrame, 
                val_df: pd.DataFrame, label2id: Dict[str, int], 
                config: Dict[str, Any]) -> Tuple[Any, Dict[str, float], Any]:
    """
    Train a transformer model using the specified training method.
    Dispatcher function that routes to the appropriate training implementation.
    
    Args:
        method: Training method ('frozen', 'finetune', 'peft')
        model_name: Name of the transformer model
        train_df, val_df: Data splits
        label2id: Label mapping
        config: Hyperparameter configuration dict
        
    Returns:
        Tuple of (model, metrics, tokenizer)
    """
    # Create args object from config
    class Args:
        def __init__(self, config: Dict[str, Any], model_name: str, method: str):
            self.model = model_name
            self.mode = method
            self.epochs = config.get('epochs', DEFAULT_EPOCHS)
            self.batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
            self.max_length = config.get('max_length', DEFAULT_MAX_LENGTH)
            self.learning_rate = config.get('learning_rate', 2e-5)
            self.loss_type = config.get('loss_type', 'standard')
            self.weight_decay = config.get('weight_decay', 0.0)
            # Additional method-specific parameters can be added here when implementing finetune/peft
    
    args = Args(config, model_name, method)
    
    # Dispatch to appropriate training function
    if method == 'frozen':
        return train_frozen_model(model_name, train_df, val_df, label2id, args)
    elif method == 'finetune':
        # TODO: Implement full fine-tuning when ready
        raise NotImplementedError(f"Training method '{method}' not yet implemented")
    elif method == 'peft':
        # TODO: Implement PEFT/LoRA when ready
        raise NotImplementedError(f"Training method '{method}' not yet implemented")
    else:
        raise ValueError(f"Unknown training method: {method}")


def run_randomized_search(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          label2id: Dict[str, int], 
                          models: list, method: str, 
                          num_combinations: int = 10,
                          hyperparameter_grid: Optional[Dict[str, list]] = None,
                          random_seed: Optional[int] = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run randomized hyperparameter search for specified models and training method.
    
    Each model receives the same hyperparameter combinations for fair comparison.
    Tracks the overall winner across all models and configurations.
    
    Args:
        train_df: Training data
        val_df: Validation data (used for evaluation during search)
        label2id: Label mapping
        models: List of model names to compare
        method: Training method ('frozen', 'finetune', 'peft') - single value only
        num_combinations: Number of random hyperparameter combinations to try per model
        hyperparameter_grid: Optional custom hyperparameter grid. If None, uses default for method
        random_seed: Random seed for reproducibility (Controls which hyperparameter combinations 
                    are randomly selected. Same seed = same combinations = reproducible results).
                    All models receive identical combinations for fair comparison.
    
    Returns:
        Tuple of (results_df, winner_config) where winner_config contains best model+config
    """
    print(f"🤖 Models: {models}")
    print(f"🔧 Method: {method}")
    print(f"🎯 Random combinations per model: {num_combinations}")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        print(f"🎲 Random seed: {random_seed}")
    
    # Select hyperparameter grid
    if hyperparameter_grid is not None:
        param_grid = hyperparameter_grid
        print(f"📋 Using custom hyperparameter grid: {list(param_grid.keys())}")
    elif method in DEFAULT_HYPERPARAMETER_GRIDS:
        param_grid = DEFAULT_HYPERPARAMETER_GRIDS[method]
        print(f"📋 Using default hyperparameter grid for '{method}' method")
    else:
        raise ValueError(f"No hyperparameter grid defined for method '{method}'. "
                        f"Please provide a custom grid or implement default grid for this method.")
    
    # Generate random hyperparameter combinations
    # Same combinations will be used for all models for fair comparison
    param_names = list(param_grid.keys())
    param_combinations = []
    
    for _ in range(num_combinations):
        config = {param: random.choice(param_grid[param]) for param in param_names}
        config['max_length'] = DEFAULT_MAX_LENGTH  # Always use default
        param_combinations.append(config)
    
    # Total number of experiments
    total_experiments = len(models) * num_combinations
    print(f"📊 Total experiments: {total_experiments}")
    print(f"⏱️  Estimated time: {total_experiments * 3:.0f}-{total_experiments * 6:.0f} minutes")
    
    results = []
    winner_config = {'val_micro_f1': 0.0}  # Track overall winner
    experiment_num = 0
    
    # Run experiments for each model with each configuration
    for model_name in models:
        print(f"\n🚀 Starting experiments for model: {model_name}")
        
        for i, config in enumerate(param_combinations, 1):
            experiment_num += 1
            print(f"\n{'='*100}")
            print(f"📋 Experiment {experiment_num}/{total_experiments}")
            print(f"   Model: {model_name}")
            print(f"   Config {i}/{num_combinations}: {config}")
            print(f"{'='*100}")
            
            try:
                start_time = time.time()
                model, metrics, tokenizer = train_model(
                    method, model_name, train_df, val_df, label2id, config
                )
                
                # Store results
                result = {
                    'experiment': experiment_num, # Experiment identification
                    'model': model_name,
                    'method': method,
                    # Hyperparameters
                    'learning_rate': config.get('learning_rate', None),
                    'batch_size': config.get('batch_size', None),
                    'epochs': config.get('epochs', None),
                    'loss_type': config.get('loss_type', None),
                    'weight_decay': config.get('weight_decay', None),
                    'max_length': config.get('max_length', None),
                    # Validation set metrics
                    'val_micro_f1': metrics.get('val_micro_f1', None),
                    'val_macro_f1': metrics.get('val_macro_f1', None),
                    'val_accuracy': metrics.get('val_accuracy', None),
                    # Timing and status
                    'training_time': time.time() - start_time,
                    'status': 'success',
                    'is_winner': False  # Will be updated after all experiments complete
                }
                results.append(result)
                
                print(f"✅ Experiment {experiment_num} completed: Val Micro-F1: {metrics['val_micro_f1']:.4f}")
                
                # Check if this is the new winner (using validation micro-F1)
                if metrics['val_micro_f1'] > winner_config.get('val_micro_f1', 0.0):
                    winner_config = {
                        'model': model_name,
                        'method': method,
                        'config': config,
                        'metrics': metrics,
                        'trained_model': model,
                        'tokenizer': tokenizer,
                        **config,
                        **metrics
                    }
                    print(f"🏆 NEW WINNER! {model_name} with Val Micro-F1: {metrics['val_micro_f1']:.4f}")
                
            except Exception as e:
                print(f"❌ Experiment {experiment_num} failed: {e}") # Store failed result with all obligatory columns (nulls for metrics)
                result = {
                    'experiment': experiment_num,  # Experiment identification
                    'model': model_name,
                    'method': method,
                    # Hyperparameters
                    'learning_rate': config.get('learning_rate', None),
                    'batch_size': config.get('batch_size', None),
                    'epochs': config.get('epochs', None),
                    'loss_type': config.get('loss_type', None),
                    'weight_decay': config.get('weight_decay', None),
                    'max_length': config.get('max_length', None),
                    # Validation set metrics (null for failed experiments)
                    'val_micro_f1': None,
                    'val_macro_f1': None,
                    'val_accuracy': None,
                    # Timing and status
                    'training_time': 0.0,
                    'status': 'failed',
                    'is_winner': False,  # Failed experiments can't be winners
                    # Optional: error message (not in obligatory columns)
                    'error': str(e)
                }
                results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Mark the winner experiment (highest val_micro_f1)
    if len(results_df) > 0 and 'val_micro_f1' in results_df.columns:
        successful = results_df[results_df['status'] == 'success']
        if len(successful) > 0 and successful['val_micro_f1'].notna().any():
            winner_idx = successful['val_micro_f1'].idxmax()
            results_df.at[winner_idx, 'is_winner'] = True
            print(f"\n🏆 Winner: Experiment {results_df.loc[winner_idx, 'experiment']} "
                  f"(Val Micro-F1: {results_df.loc[winner_idx, 'val_micro_f1']:.4f})")
    
    # TODO: Calculate metrics scores on the training set of the winning model as well, to check for further analysis and overfitting analysis. 

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/metrics/transformers/randomized_search_{method}_{timestamp}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    
    # Print comprehensive analysis
    print_randomized_search_results(results_df, winner_config, results_file)
    
    # TODO: Create notebook compare frozen vs finetune vs peft best results
    # TODO: Add cross-method comparison plots (performance vs training time, method strengths/weaknesses) (internal comparison - between transformers ; external comparison - between transformers and shallow models)
    
    return results_df, winner_config


def print_randomized_search_results(results_df: pd.DataFrame, winner_config: Dict[str, Any], 
                                    results_file: str) -> None:
    """Print comprehensive results analysis for randomized search."""
    print(f"\n{'='*100}")
    print("🎯 RANDOMIZED SEARCH RESULTS")
    print(f"{'='*100}")
    
    # Overall winner
    print(f"\n🥇 OVERALL WINNER:")
    print(f"   Model: {winner_config['model']}")
    print(f"   Method: {winner_config['method']}")
    print(f"   Val Micro-F1: {winner_config.get('val_micro_f1', 0):.4f}")
    print(f"   Val Macro-F1: {winner_config.get('val_macro_f1', 0):.4f}")
    print(f"   Val Accuracy: {winner_config.get('val_accuracy', 0):.4f}")
    print(f"\n   Hyperparameters:")
    for key in ['learning_rate', 'batch_size', 'epochs', 'loss_type', 'weight_decay']:
        if key in winner_config:
            print(f"     {key}: {winner_config[key]}")
    
    # Top 10 configurations
    successful_results = results_df[results_df['status'] == 'success']
    if len(successful_results) > 0:
        top_results = successful_results.sort_values('val_micro_f1', ascending=False).head(10)
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        display_cols = ['model', 'learning_rate', 'batch_size', 'loss_type', 'val_micro_f1', 'val_accuracy']
        if all(col in top_results.columns for col in display_cols):
            print(top_results[display_cols].to_string(index=False))
    
    # Analysis by model
    print(f"\n📊 ANALYSIS BY MODEL:")
    for model in results_df['model'].unique():
        model_results = successful_results[successful_results['model'] == model]
        if len(model_results) > 0 and 'val_micro_f1' in model_results.columns:
            best_idx = model_results['val_micro_f1'].idxmax()
            best = model_results.loc[best_idx]
            avg_f1 = model_results['val_micro_f1'].mean()
            std_f1 = model_results['val_micro_f1'].std()
            print(f"   {model}:")
            print(f"     Best Val Micro-F1: {best['val_micro_f1']:.4f} (LR: {best['learning_rate']}, BS: {best['batch_size']})")
            print(f"     Avg Val Micro-F1: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Hyperparameter analysis (only for params that exist in results)
    print(f"\n📈 HYPERPARAMETER ANALYSIS:")
    
    for param in ['learning_rate', 'batch_size', 'loss_type']:
        if param in successful_results.columns and 'val_micro_f1' in successful_results.columns:
            print(f"\n{param.replace('_', ' ').title()}:")
            analysis = successful_results.groupby(param)['val_micro_f1'].agg(['mean', 'max', 'std', 'count']).round(4)
            print(analysis.to_string())
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"{'='*100}\n")


def save_winner_model(winner_config: Dict[str, Any], val_df: pd.DataFrame, 
                     label2id: Dict[str, int]) -> None:
    """
    Save the overall winner model from randomized search.
    
    Args:
        winner_config: Dictionary containing winner model, config, and metrics
        val_df: Validation dataframe for generating confusion matrix
        label2id: Label mapping
    """
    if 'trained_model' not in winner_config or 'tokenizer' not in winner_config:
        print("⚠️ Warning: No trained model found in winner config, skipping save")
        return
    
    print("\n💾 Saving winner model...")
    
    model = winner_config['trained_model']
    tokenizer = winner_config['tokenizer']
    metrics = winner_config['metrics']
    method = winner_config['method']
    
    # Create unique run_id for winner
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_clean = winner_config['model'].replace('/', '_')
    run_id = f"{model_name_clean}_{method}_winner_{timestamp}"
    
    # Save metrics
    metrics_file = f"reports/metrics/transformers/{run_id}.json"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    # Add config info to metrics
    full_metrics = {
        **metrics,
        'model_name': winner_config['model'],
        'training_method': method,
        'hyperparameters': {k: v for k, v in winner_config.items() 
                          if k not in ['trained_model', 'tokenizer', 'metrics', 'config']},
        'run_id': run_id,
        'timestamp': timestamp
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    print(f"📊 Saved winner metrics to {metrics_file}\n")
    
    # Check if we should update the winner checkpoint (compare with existing winner, if any)
    winner_dir = f"models/transformers/{method}_winner"
    winner_info_file = f"{winner_dir}/model_info.json"
    should_save_checkpoint = True
    
    if os.path.exists(winner_info_file):
        try:
            with open(winner_info_file, 'r') as f:
                existing_winner = json.load(f)
            existing_f1 = existing_winner.get('val_micro_f1', 0.0)
            new_f1 = metrics['val_micro_f1']
            
            if new_f1 > existing_f1:
                print(f"🎉 New winner beats existing winner! ({new_f1:.4f} > {existing_f1:.4f})")
                should_save_checkpoint = True
            else:
                print(f"⚠️  New winner ({new_f1:.4f}) does NOT beat existing winner ({existing_f1:.4f})")
                print(f"   Keeping existing checkpoint in {winner_dir}")
                should_save_checkpoint = False
        except Exception as e:
            print(f"⚠️  Could not read existing winner info: {e}")
            print(f"   Will overwrite with new winner")
            should_save_checkpoint = True
    else:
        print(f"✨ No existing winner found - saving first winner checkpoint")
        should_save_checkpoint = True
    
    # Save model checkpoint only if it's better than existing
    if should_save_checkpoint:
        os.makedirs(winner_dir, exist_ok=True)
        
        # Check if model has save_pretrained method
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(winner_dir)
            tokenizer.save_pretrained(winner_dir)
            print(f"🏆 Saved winner model checkpoint to {winner_dir}")
        else:
            # For custom models, save the state dict
            torch.save(model.state_dict(), f"{winner_dir}/pytorch_model.bin")
            tokenizer.save_pretrained(winner_dir)
        
        # Always save model info (for comparison in future runs)
        model_info = {
            'model_type': type(model).__name__,
            'model_name': winner_config['model'],
            'training_method': winner_config['method'],
            **full_metrics
        }
        with open(f"{winner_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Updated winner checkpoint with Val Micro-F1: {metrics['val_micro_f1']:.4f}")
    
    # Generate and save confusion matrix
    print("📈 Generating confusion matrix for winner...")
    
    # Get predictions on validation set
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Tokenize validation data
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    # Create dataset and loader
    from torch.utils.data import TensorDataset
    encodings = tokenizer(val_texts, truncation=True, padding=True, 
                         max_length=DEFAULT_MAX_LENGTH, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], 
                           torch.tensor(val_labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad(): # Don't compute gradients (faster, saves memory)
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']  # Extract logits from the output dictionary
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())  # Convert to cpu + numpy array to allow scikit-learn evaluation
            all_labels.extend(labels.cpu().numpy())  # Convert to cpu + numpy array to allow scikit-learn evaluation
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label2id.keys()),
                yticklabels=list(label2id.keys()))
    plt.title(f'Confusion Matrix - Winner Model\n{winner_config["model"]} ({method})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plot_file = f"reports/plots/transformers/{run_id}_cm_val.png"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved confusion matrix to {plot_file}")
    print("✅ Winner model saved successfully!")


# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Function ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train transformer models for emotion classification with randomized hyperparameter search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single model with 10 random hyperparameter combinations
  python scripts/5_train_transformer.py --model distilbert-base-uncased --mode frozen --num_combinations 10
  
  # Multiple models (each gets same random combinations for fair comparison)
  python scripts/5_train_transformer.py --model distilbert-base-uncased bert-base-uncased --mode frozen --num_combinations 10
  
  # Custom hyperparameter grid (as JSON string)
  python scripts/5_train_transformer.py --model distilbert-base-uncased --mode frozen --num_combinations 5 \\
      --hyperparameter_grid '{"learning_rate": [0.0001, 0.0005], "batch_size": [8, 16], "epochs": [3, 5], "loss_type": ["weighted"]}'
        """
    )
    
    # Model and method arguments
    parser.add_argument('--model', nargs='+', default=['distilbert-base-uncased'],
                       choices=['distilbert-base-uncased', 'bert-base-uncased'],
                       help='One or more transformer models to compare (space-separated for multiple)')
    parser.add_argument('--mode', type=str, default='frozen',
                       choices=['frozen', 'finetune', 'peft'],
                       help='Training method (single value only): frozen (currently available), finetune, or peft')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default=DEFAULT_TRAIN_DATA,
                       help='Path to training data file')
    parser.add_argument('--val_data', type=str, default=DEFAULT_VAL_DATA,
                       help='Path to validation data file')
    parser.add_argument('--test_data', type=str, default=DEFAULT_TEST_DATA,
                       help='Path to test data file')
    parser.add_argument('--labels_file', type=str, default=DEFAULT_LABELS_FILE,
                       help='Path to labels file')
    
    # Randomized search configuration
    parser.add_argument('--num_combinations', type=int, default=10,
                       help='Number of random hyperparameter combinations to try per model')
    parser.add_argument('--hyperparameter_grid', type=str, default=None,
                       help='Custom hyperparameter grid as JSON string. If not provided, uses default grid for the method. '
                            'Format: {"learning_rate": [0.0001, 0.0005], "batch_size": [8, 16], ...}')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility. Controls which hyperparameter combinations are selected. '
                            'Same seed ensures same combinations across runs.')
    
    args = parser.parse_args()
    
    # Parse and validate hyperparameter grid early (fail fast)
    hyperparameter_grid = None
    if args.hyperparameter_grid:
        try:
            hyperparameter_grid = json.loads(args.hyperparameter_grid)
            print(f"Loaded custom hyperparameter grid: {list(hyperparameter_grid.keys())}")
        except json.JSONDecodeError as e:
            print(f"Error parsing hyperparameter_grid JSON: {e}")
            sys.exit(1)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️ Warning: No GPU detected. Training will be slow on CPU.")
    
    # Fail-fast checks
    print("\n🔍 Performing fail-fast checks...")
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
    
    # Run randomized hyperparameter search
    # Works for both single and multiple models (test_df reserved for final evaluation)
    num_models = len(args.model)
    print(f"\n🎲 Starting randomized search with {num_models} model(s) × {args.num_combinations} combinations")
    
    try:
        results_df, winner_config = run_randomized_search(
            train_df, val_df, label2id,
            models=args.model,
            method=args.mode,
            num_combinations=args.num_combinations,
            hyperparameter_grid=hyperparameter_grid,
            random_seed=args.random_seed
        )
        
        # Save the winner model
        save_winner_model(winner_config, val_df, label2id)
        
        print("\n🎉 Randomized search completed successfully!")
        print(f"🏆 Winner: {winner_config['model']} with Val Micro-F1: {winner_config.get('val_micro_f1', 0):.4f}\n")
        
        # TODO: Do the followings:
        # TODO: make sure it's consistent with shallow models implementation
        # TODO: Retrain the winner model on the train&val sets 
        # TODO: Evaluate on test set
        # TODO: Save the winner model on the test set
        # TODO: Generate test set confusion matrix and metrics for winner (and save)
        # TODO: Compare test vs validation performance to check for overfitting

    # TODO: "frozen" method - try to improve performance
        
    except Exception as e:
        print(f"❌ Error during randomized search: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
