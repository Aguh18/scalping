"""
Lightweight training script for slower PCs
Uses sample data and optimized parameters for faster training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
from datetime import datetime

from .merge_data import load_and_merge_2024_data
from .features import calculate_technical_indicators
from .labeling import calculate_tp_sl_labels


def prepare_lightweight_data(sample_size=50000):
    """
    Prepare data with sampling for faster training
    """
    print("Loading and preparing lightweight data...")
    
    # Load data
    df = load_and_merge_2024_data()
    print(f"Original dataset shape: {df.shape}")
    
    # Sample data for faster training
    if len(df) > sample_size:
        # Take recent data (more relevant for current market)
        df_sample = df.tail(sample_size).copy()
        print(f"Using sample of {sample_size} recent bars")
    else:
        df_sample = df.copy()
        print(f"Using full dataset: {len(df_sample)} bars")
    
    # Calculate features
    print("Calculating technical indicators...")
    df_features = calculate_technical_indicators(df_sample)
    
    # Calculate labels
    print("Calculating TP/SL labels...")
    df_labeled = calculate_tp_sl_labels(df_features)
    
    # Remove rows with NaN values
    df_clean = df_labeled.dropna()
    print(f"Clean dataset shape: {df_clean.shape}")
    
    # Prepare features and labels
    feature_columns = [col for col in df_clean.columns 
                      if col not in ['entry', 'tp', 'sl', 'signal']]
    
    X = df_clean[feature_columns].values
    y = (df_clean['signal'] == 'Long').astype(int)  # Binary: Long=1, Short/None=0
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (time series split)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_columns, scaler


def train_lightweight_model():
    """
    Train lightweight XGBoost model (no LSTM for speed)
    """
    print("Training lightweight model...")
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_columns, scaler = prepare_lightweight_data()
    
    # Train XGBoost with optimized parameters for speed
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,  # Reduced from default
        max_depth=6,       # Reduced for speed
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,         # Use all CPU cores
        tree_method='hist'  # Faster training
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Save model and results
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/lightweight_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = 'models/lightweight_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature columns
    feature_path = 'models/lightweight_features.pkl'
    joblib.dump(feature_columns, feature_path)
    print(f"Features saved to: {feature_path}")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'model_type': 'lightweight_xgboost',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = 'models/lightweight_results.pkl'
    joblib.dump(results, results_path)
    print(f"Results saved to: {results_path}")
    
    return model, scaler, feature_columns, results


def test_lightweight_signal():
    """
    Test signal generation with lightweight model
    """
    print("Testing lightweight signal generation...")
    
    try:
        # Load model components
        model = joblib.load('models/lightweight_model.pkl')
        scaler = joblib.load('models/lightweight_scaler.pkl')
        feature_columns = joblib.load('models/lightweight_features.pkl')
        
        # Load recent data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        df_labeled = calculate_tp_sl_labels(df_features)
        
        # Get latest data point
        latest = df_labeled.iloc[-1:].copy()
        
        # Prepare features
        X_latest = latest[feature_columns].values
        X_scaled = scaler.transform(X_latest)
        
        # Generate prediction
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0][1]
        
        # Generate signal
        if prediction == 1:
            signal = "Long"
        else:
            signal = "Short"
        
        result = {
            "signal": signal,
            "entry": float(latest['entry'].iloc[0]),
            "tp": float(latest['tp'].iloc[0]),
            "sl": float(latest['sl'].iloc[0]),
            "confidence": float(confidence)
        }
        
        print("✅ Signal generated successfully:")
        print(f"Signal: {result['signal']}")
        print(f"Entry: {result['entry']:.2f}")
        print(f"TP: {result['tp']:.2f}")
        print(f"SL: {result['sl']:.2f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating signal: {e}")
        return None


if __name__ == "__main__":
    # Train lightweight model
    model, scaler, features, results = train_lightweight_model()
    
    # Test signal generation
    test_lightweight_signal()
