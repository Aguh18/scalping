"""
Quick Training Script - Optimized for slower PCs
Uses XGBoost only (no LSTM) for faster training while maintaining accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import os
import glob
from datetime import datetime

# Import our modules
from src.merge_data import load_and_merge_2024_data
from src.features import calculate_technical_indicators
from src.labeling import calculate_tp_sl_labels


def quick_train():
    """
    Quick training with optimized parameters for speed
    """
    print("🚀 Quick Training - Optimized for Speed")
    print("=" * 50)
    
    # Step 1: Load data (sample for speed)
    print("\n📊 Step 1: Loading data...")
    df = load_and_merge_2024_data()
    print(f"Original dataset: {df.shape[0]} bars")
    
    # Use recent 30,000 bars for faster training
    sample_size = min(30000, len(df))
    df_sample = df.tail(sample_size).copy()
    print(f"Using sample: {sample_size} recent bars")
    
    # Step 2: Calculate features
    print("\n🔧 Step 2: Calculating features...")
    df_features = calculate_technical_indicators(df_sample)
    
    # Step 3: Calculate labels
    print("\n🏷️ Step 3: Calculating labels...")
    df_labeled = calculate_tp_sl_labels(df_features)
    
    # Clean data
    df_clean = df_labeled.dropna()
    print(f"Clean dataset: {df_clean.shape[0]} bars")
    
    # Step 4: Prepare features
    print("\n⚙️ Step 4: Preparing features...")
    feature_columns = [col for col in df_clean.columns 
                      if col not in ['entry', 'tp', 'sl', 'signal']]
    
    X = df_clean[feature_columns].values
    y = (df_clean['signal'] == 'Long').astype(int)
    
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Step 5: Train model
    print("\n🏋️ Step 5: Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=50,      # Reduced for speed
        max_depth=4,          # Reduced for speed
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,            # Use all CPU cores
        tree_method='hist'     # Faster training
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate
    print("\n📈 Step 6: Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-Score: {f1_score:.4f}")
    
    # Step 7: Save model
    print("\n💾 Step 7: Saving model...")
    os.makedirs('models', exist_ok=True)
    
    # Save components
    joblib.dump(model, 'models/quick_model.pkl')
    joblib.dump(scaler, 'models/quick_scaler.pkl')
    joblib.dump(feature_columns, 'models/quick_features.pkl')
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'model_type': 'quick_xgboost',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(results, 'models/quick_results.pkl')
    
    print("✅ Model saved successfully!")
    return model, scaler, feature_columns, results


def quick_test():
    """
    Quick test of signal generation
    """
    print("\n🎯 Testing signal generation...")
    
    try:
        # Load model
        model = joblib.load('models/quick_model.pkl')
        scaler = joblib.load('models/quick_scaler.pkl')
        feature_columns = joblib.load('models/quick_features.pkl')
        
        # Load recent data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        df_labeled = calculate_tp_sl_labels(df_features)
        
        # Get latest data
        latest = df_labeled.iloc[-1:].copy()
        
        # Prepare features
        X_latest = latest[feature_columns].values
        X_scaled = scaler.transform(X_latest)
        
        # Generate prediction
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0][1]
        
        # Generate signal
        signal = "Long" if prediction == 1 else "Short"
        
        result = {
            "signal": signal,
            "entry": float(latest['entry'].iloc[0]),
            "tp": float(latest['tp'].iloc[0]),
            "sl": float(latest['sl'].iloc[0]),
            "confidence": float(confidence)
        }
        
        print("✅ Signal generated:")
        print(f"   Signal: {result['signal']}")
        print(f"   Entry: {result['entry']:.2f}")
        print(f"   TP: {result['tp']:.2f}")
        print(f"   SL: {result['sl']:.2f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Quick train
    model, scaler, features, results = quick_train()
    
    # Quick test
    quick_test()
    
    print("\n🎉 Quick training completed!")
    print("📊 Model ready for trading!")
