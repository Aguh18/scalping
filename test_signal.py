"""
Test Signal Generation Script
"""

import pandas as pd
import numpy as np
import joblib
from src.merge_data import load_and_merge_2024_data
from src.features import calculate_technical_indicators
from src.labeling import calculate_tp_sl_labels


def test_signal():
    """
    Test signal generation with trained model
    """
    print("🎯 Testing Signal Generation")
    print("=" * 40)
    
    try:
        # Load model components
        print("📥 Loading model...")
        model = joblib.load('models/quick_model.pkl')
        scaler = joblib.load('models/quick_scaler.pkl')
        feature_columns = joblib.load('models/quick_features.pkl')
        
        print(f"✅ Model loaded with {len(feature_columns)} features")
        
        # Load recent data
        print("\n📊 Loading recent data...")
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        df_labeled = calculate_tp_sl_labels(df_features)
        
        # Get latest 5 data points
        latest_data = df_labeled.tail(5).copy()
        
        print(f"✅ Loaded {len(latest_data)} recent data points")
        
        # Generate signals for recent data
        print("\n🎯 Generating signals...")
        
        for i, (idx, row) in enumerate(latest_data.iterrows()):
            try:
                # Prepare features
                X_latest = row[feature_columns].values.reshape(1, -1)
                X_scaled = scaler.transform(X_latest)
                
                # Generate prediction
                prediction = model.predict(X_scaled)[0]
                confidence = model.predict_proba(X_scaled)[0][1]
                
                # Generate signal
                signal = "Long" if prediction == 1 else "Short"
                
                print(f"\n📈 Signal {i+1} (Time: {idx}):")
                print(f"   Signal: {signal}")
                print(f"   Entry: {row['entry']:.2f}")
                print(f"   TP: {row['tp']:.2f}")
                print(f"   SL: {row['sl']:.2f}")
                print(f"   Confidence: {confidence:.4f}")
                
            except Exception as e:
                print(f"❌ Error generating signal {i+1}: {e}")
                continue
        
        # Generate latest signal
        print(f"\n🎯 Latest Signal:")
        latest = latest_data.iloc[-1]
        
        X_latest = latest[feature_columns].values.reshape(1, -1)
        X_scaled = scaler.transform(X_latest)
        
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0][1]
        
        signal = "Long" if prediction == 1 else "Short"
        
        result = {
            "signal": signal,
            "entry": float(latest['entry']),
            "tp": float(latest['tp']),
            "sl": float(latest['sl']),
            "confidence": float(confidence),
            "timestamp": str(latest.name)
        }
        
        print(f"   Signal: {result['signal']}")
        print(f"   Entry: {result['entry']:.2f}")
        print(f"   TP: {result['tp']:.2f}")
        print(f"   SL: {result['sl']:.2f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Time: {result['timestamp']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    result = test_signal()
    
    if result:
        print(f"\n🎉 Signal generation successful!")
        print(f"📊 Model is ready for trading!")
    else:
        print(f"\n❌ Signal generation failed!")
