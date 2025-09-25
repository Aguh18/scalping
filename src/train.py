"""
Training module for the hybrid LSTM + XGBoost model
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from merge_data import load_and_merge_2024_data
from features import calculate_technical_indicators, prepare_features_for_model, normalize_features
from labeling import calculate_tp_sl_labels, create_trading_signals
from model import HybridLSTMXGBModel, create_sequences_with_labels, split_time_series_data


def prepare_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare all training data
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_columns)
    """
    print("Loading and preparing data...")
    
    # Load data
    df = load_and_merge_2024_data()
    print(f"Loaded data shape: {df.shape}")
    
    # Calculate technical indicators
    df_features = calculate_technical_indicators(df)
    print(f"Features calculated, shape: {df_features.shape}")
    
    # Create labels
    df_labeled = calculate_tp_sl_labels(df_features)
    signals = create_trading_signals(df_labeled)
    print(f"Labels created, shape: {signals.shape}")
    
    # Prepare features
    feature_df, feature_columns = prepare_features_for_model(df_features)
    normalized_df, scaler = normalize_features(feature_df)
    
    # Align data
    common_index = normalized_df.index.intersection(signals.index)
    X_data = normalized_df.loc[common_index].values
    y_data = signals.loc[common_index]['label'].values
    
    print(f"Aligned data shapes - X: {X_data.shape}, y: {y_data.shape}")
    print(f"Label distribution: {np.bincount(y_data.astype(int))}")
    
    # Create sequences
    X, y = create_sequences_with_labels(X_data, y_data, sequence_length=60)
    print(f"Sequences created - X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(X, y)
    
    print(f"Data split completed:")
    print(f"Train: X{X_train.shape}, y{y_train.shape}")
    print(f"Val: X{X_val.shape}, y{y_val.shape}")
    print(f"Test: X{X_test.shape}, y{y_test.shape}")
    
    # Save scaler
    os.makedirs("./models", exist_ok=True)
    joblib.dump(scaler, "./models/scaler.pkl")
    joblib.dump(feature_columns, "./models/feature_columns.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 128) -> HybridLSTMXGBModel:
    """
    Train the hybrid model
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trained model
    """
    print("Initializing hybrid model...")
    
    # Create model
    model = HybridLSTMXGBModel(
        sequence_length=60,
        n_features=X_train.shape[2],
        hidden_units=128,
        dropout_rate=0.2
    )
    
    # Train model
    print("Starting training...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_path="./models/hybrid_model.h5"
    )
    
    print("Training completed!")
    return model, history


def evaluate_model(model: HybridLSTMXGBModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test labels
        
    Returns:
        Evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return metrics


def plot_training_history(history: Dict[str, Any], save_path: str = "./models/training_history.png"):
    """
    Plot training history
    
    Args:
        history: Training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Training Precision')
    axes[1, 0].plot(history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Training Recall')
    axes[1, 1].plot(history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "./models/confusion_matrix.png"):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Trade/Short', 'Long'], 
                yticklabels=['No Trade/Short', 'Long'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix plot saved to {save_path}")


def main():
    """Main training function"""
    try:
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_columns = prepare_training_data()
        
        # Train model
        model, history = train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=128)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Plot results
        plot_training_history(history)
        plot_confusion_matrix(metrics['confusion_matrix'])
        
        # Save evaluation results
        results = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'feature_columns': feature_columns
        }
        
        joblib.dump(results, "./models/evaluation_results.pkl")
        print("Evaluation results saved to ./models/evaluation_results.pkl")
        
        print("\nTraining completed successfully!")
        print("Model files saved in ./models/ directory")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
