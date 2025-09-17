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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
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
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer encoder - SAVES COMPUTATION TIME
        print(f"🧊 Freezing transformer encoder parameters...")
        for param in self.transformer.parameters():
            param.requires_grad = False
        print(f"✅ Frozen {sum(1 for p in self.transformer.parameters())} parameters")
        
        # Classification head - start with linear head
        # If needed, try MLP (Linear→GELU→Dropout→Linear)
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
        print(f"✅ Model saved successfully")

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
    
    # Setup optimizer - ONLY trains classification head (faster)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🎯 Training {len(trainable_params)} parameters (head only)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    print(f"🎯 Using learning rate: {args.learning_rate}")
    
    # Training loop with early stopping
    model.train()
    best_val_f1 = 0
    patience_counter = 0
    early_stopping_patience = 3
    
    print(f"🏃‍♂️ Starting training for {args.epochs} epochs...")
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
    
    total_time = time.time() - start_time
    training_time = time.time() - training_start
    
    metrics = {
        'model': model_name,
        'mode': args.mode,
        'accuracy': accuracy,
        'micro_f1': micro_f1,  # PRIMARY metric for imbalanced data
        'macro_f1': macro_f1,  # Shows per-class performance
        'weighted_f1': weighted_f1,  # Weighted by class frequency
        'loss': avg_val_loss,
        'epochs_trained': epoch + 1,
        'total_time_seconds': total_time,
        'training_time_seconds': training_time,
        'classification_report': classification_report(
            final_labels, final_predictions, 
            target_names=list(label2id.keys()),
            output_dict=True
        )
    }
    
    print(f"✅ Training completed in {total_time:.1f}s (training: {training_time:.1f}s)")
    print(f"📊 Final Metrics - Accuracy: {accuracy:.3f}, Micro-F1: {micro_f1:.3f}, Macro-F1: {macro_f1:.3f}")
    
    return model, metrics, tokenizer

def save_results(model, tokenizer, metrics: Dict[str, Any], model_name: str, mode: str, 
                val_df: pd.DataFrame, label2id: Dict[str, int]):
    """Save metrics, plots, and checkpoints - OPTIMIZED file operations."""
    
    print("💾 Saving results...")
    save_start = time.time()
    
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
        print(f"🏆 Saved winner checkpoint to {winner_dir}")
    
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
    
    save_time = time.time() - save_start
    print(f"📈 Saved confusion matrix to {plot_file}")
    print(f"✅ Results saved in {save_time:.1f}s")

def run_hyperparameter_comparison(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                 test_df: pd.DataFrame, label2id: Dict[str, int], 
                                 models_to_compare: list, quick_mode: bool = False, 
                                 mode: str = 'frozen') -> pd.DataFrame:
    """
    Run comprehensive hyperparameter comparison with multiple models.
    Similar to randomized search but systematic for reproducibility.
    Tests different combinations and saves the best results.
    
    Args:
        train_df, val_df, test_df: Data splits
        label2id: Label mapping
        models_to_compare: List of model names to compare
        quick_mode: If True, run fewer combinations for faster testing
        
    Returns:
        DataFrame with all experiment results
    """
    print("🔬 Starting comprehensive hyperparameter comparison...")
    print(f"🤖 Models to compare: {models_to_compare}")
    
    # Define comprehensive hyperparameter grid
    if quick_mode:
        # Quick mode: fewer combinations for testing
        learning_rates = [2e-5]
        batch_sizes = [8, 16]
        loss_types = ['weighted']
        epochs = 2
        print("⚡ Quick mode: Testing essential combinations only")
    else:
        # Full mode: comprehensive search
        learning_rates = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5]  # More LR options
        batch_sizes = [8, 16, 32]  # Different batch sizes
        loss_types = ['weighted', 'standard']  # Compare loss functions
        epochs = 4  # More epochs for better convergence
        print("🔍 Full mode: Comprehensive hyperparameter search")
    
    # Generate all combinations
    configs = []
    for model in models_to_compare:
        for lr in learning_rates:
            for bs in batch_sizes:
                for loss in loss_types:
                    configs.append({
                        'model': model,
                        'lr': lr,
                        'bs': bs,
                        'loss': loss,
                        'epochs': epochs
                    })
    
    print(f"📊 Total experiments: {len(configs)}")
    print(f"⏱️ Estimated time: {len(configs) * 3:.0f}-{len(configs) * 5:.0f} minutes")
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n📋 Experiment {i}/{len(configs)}: {config}")
        
        # Create args object
        class Args:
            def __init__(self, config, mode):
                self.model = config['model']
                self.mode = mode  # Use the passed mode
                self.epochs = config['epochs']
                self.batch_size = config['bs']
                self.max_length = 512
                self.learning_rate = config['lr']
                self.loss_type = config['loss']
        
        args = Args(config, mode)
        
        try:
            start_time = time.time()
            model, metrics, tokenizer = train_frozen_model(args.model, train_df, val_df, label2id, args)
            
            # Store results
            result = {
                'experiment': i,
                'model': config['model'],
                'learning_rate': config['lr'],
                'batch_size': config['bs'],
                'loss_type': config['loss'],
                'epochs': config['epochs'],
                'micro_f1': metrics['micro_f1'],
                'macro_f1': metrics['macro_f1'],
                'accuracy': metrics['accuracy'],
                'training_time': time.time() - start_time
            }
            results.append(result)
            
            print(f"✅ Experiment {i} completed: Micro-F1: {metrics['micro_f1']:.3f}")
            
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            results.append({
                'experiment': i,
                'model': config['model'],
                'learning_rate': config['lr'],
                'batch_size': config['bs'],
                'loss_type': config['loss'],
                'epochs': config['epochs'],
                'micro_f1': 0.0,
                'macro_f1': 0.0,
                'accuracy': 0.0,
                'training_time': 0.0,
                'error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_file = f"reports/metrics/transformers/hyperparameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    
    # Comprehensive results analysis
    print(f"\n🎯 HYPERPARAMETER COMPARISON RESULTS:")
    print("=" * 100)
    
    # Show top 10 results
    top_results = results_df.sort_values('micro_f1', ascending=False).head(10)
    print("🏆 TOP 10 CONFIGURATIONS:")
    print(top_results[['model', 'learning_rate', 'batch_size', 'loss_type', 'micro_f1', 'accuracy', 'macro_f1']].to_string(index=False))
    
    # Best overall result
    best_result = results_df.loc[results_df['micro_f1'].idxmax()]
    print(f"\n🥇 BEST OVERALL CONFIGURATION:")
    print(f"   Model: {best_result['model']}")
    print(f"   Learning Rate: {best_result['learning_rate']}")
    print(f"   Batch Size: {best_result['batch_size']}")
    print(f"   Loss Type: {best_result['loss_type']}")
    print(f"   Micro-F1: {best_result['micro_f1']:.4f}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Macro-F1: {best_result['macro_f1']:.4f}")
    print(f"   Training Time: {best_result['training_time']:.1f}s")
    
    # Analysis by model
    print(f"\n📊 ANALYSIS BY MODEL:")
    for model in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model]
        best_model_result = model_results.loc[model_results['micro_f1'].idxmax()]
        avg_f1 = model_results['micro_f1'].mean()
        print(f"   {model}:")
        print(f"     Best Micro-F1: {best_model_result['micro_f1']:.4f} (LR: {best_model_result['learning_rate']}, BS: {best_model_result['batch_size']})")
        print(f"     Average Micro-F1: {avg_f1:.4f}")
    
    # Analysis by hyperparameters
    print(f"\n📈 ANALYSIS BY HYPERPARAMETERS:")
    print("Learning Rate Performance:")
    lr_analysis = results_df.groupby('learning_rate')['micro_f1'].agg(['mean', 'max', 'std']).round(4)
    print(lr_analysis.to_string())
    
    print("\nBatch Size Performance:")
    bs_analysis = results_df.groupby('batch_size')['micro_f1'].agg(['mean', 'max', 'std']).round(4)
    print(bs_analysis.to_string())
    
    print("\nLoss Type Performance:")
    loss_analysis = results_df.groupby('loss_type')['micro_f1'].agg(['mean', 'max', 'std']).round(4)
    print(loss_analysis.to_string())
    
    # Save detailed analysis
    analysis_file = results_file.replace('.csv', '_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write("HYPERPARAMETER COMPARISON ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Configuration:\n{best_result.to_string()}\n\n")
        f.write("Learning Rate Analysis:\n")
        f.write(lr_analysis.to_string() + "\n\n")
        f.write("Batch Size Analysis:\n")
        f.write(bs_analysis.to_string() + "\n\n")
        f.write("Loss Type Analysis:\n")
        f.write(loss_analysis.to_string() + "\n")
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"📋 Detailed analysis saved to: {analysis_file}")
    
    return results_df

# --------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------- Main Function ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train transformer models for emotion classification')
    parser.add_argument('--model', nargs='+', default=['distilbert-base-uncased'],
                       choices=['distilbert-base-uncased', 'bert-base-uncased'],
                       help='Transformer model(s) to use (space-separated for multiple)')
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
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for optimizer')
    parser.add_argument('--loss_type', type=str, default='weighted', 
                       choices=['weighted', 'standard'],
                       help='Loss function type: weighted=class weights (default), standard=no imbalance handling')
    parser.add_argument('--compare_hyperparams', action='store_true',
                       help='Run hyperparameter comparison for the specified model(s) and mode')
    parser.add_argument('--quick_comparison', action='store_true',
                       help='Run quick comparison with fewer hyperparameter combinations')
    
    args = parser.parse_args()
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️ Warning: No GPU detected. Training will be slow on CPU.")
    
    # Fail-fast checks
    print("🔍 Performing fail-fast checks...")
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
    
    # Determine what to run based on arguments
    if args.compare_hyperparams:
        # Hyperparameter comparison mode
        print(f"🔬 Running hyperparameter comparison for {args.mode} mode with models: {args.model}")
        try:
            results_df = run_hyperparameter_comparison(
                train_df, val_df, test_df, label2id, 
                models_to_compare=args.model,
                quick_mode=args.quick_comparison,
                mode=args.mode
            )
            print("🎉 Hyperparameter comparison completed successfully!")
        except Exception as e:
            print(f"❌ Error during hyperparameter comparison: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif len(args.model) > 1:
        # Multiple models with your specified hyperparameters
        print(f"🤖 Training multiple models with your hyperparameters: {args.model}")
        all_results = []
        
        for model_name in args.model:
            print(f"\n🚀 Training {model_name}...")
            try:
                # Create args for single model
                single_args = argparse.Namespace(**vars(args))
                single_args.model = model_name
                
                if args.mode == 'frozen':
                    model, metrics, tokenizer = train_frozen_model(model_name, train_df, val_df, label2id, single_args)
                    save_results(model, tokenizer, metrics, model_name, args.mode, val_df, label2id)
                    all_results.append(metrics)
                    print(f"✅ {model_name} completed: Micro-F1: {metrics['micro_f1']:.3f}")
                else:
                    print(f"❌ Mode '{args.mode}' not implemented yet for {model_name}")
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        # Show comparison
        if all_results:
            print(f"\n🏆 MULTI-MODEL COMPARISON:")
            for i, (model_name, result) in enumerate(zip(args.model, all_results)):
                print(f"   {i+1}. {model_name}: Micro-F1: {result['micro_f1']:.4f}, Accuracy: {result['accuracy']:.4f}")
        
        print("🎉 All models trained successfully!")
    else:
        # Single model training
        if args.mode == 'frozen':
            try:
                single_args = argparse.Namespace(**vars(args))
                single_args.model = args.model[0]  # Use first model
                
                model, metrics, tokenizer = train_frozen_model(args.model[0], train_df, val_df, label2id, single_args)
                save_results(model, tokenizer, metrics, args.model[0], args.mode, val_df, label2id)
                print("🎉 Training completed successfully!")
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
