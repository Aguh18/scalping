"""
Real-time signal generation module
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Tuple
import os

from model import HybridLSTMXGBModel
from features import calculate_technical_indicators, normalize_features


class SignalGenerator:
    """
    Real-time signal generator for Bitcoin scalping
    """
    
    def __init__(self, model_path: str = "./models/hybrid_model.h5"):
        """
        Initialize signal generator
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = 60
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers"""
        try:
            # Load model
            self.model = HybridLSTMXGBModel(
                sequence_length=self.sequence_length,
                n_features=50,  # Will be updated when scaler is loaded
                hidden_units=128,
                dropout_rate=0.2
            )
            self.model.load_model(self.model_path)
            
            # Load scaler and feature columns
            self.scaler = joblib.load("./models/scaler.pkl")
            self.feature_columns = joblib.load("./models/feature_columns.pkl")
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def prepare_latest_window(self, df: pd.DataFrame) -> Tuple[np.ndarray, float, float]:
        """
        Prepare the latest window for prediction
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            
        Returns:
            Tuple of (normalized_sequence, current_atr, current_price)
        """
        # Calculate technical indicators if not present
        if 'atr_14' not in df.columns:
            df = calculate_technical_indicators(df)
        
        # Get latest values
        current_atr = df['atr_14'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Prepare features
        feature_df = df[self.feature_columns].copy()
        
        # Normalize features
        normalized_df, _ = normalize_features(feature_df, self.scaler, fit=False)
        
        # Get the latest sequence
        if len(normalized_df) < self.sequence_length:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} bars")
        
        sequence = normalized_df.iloc[-self.sequence_length:].values
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        return sequence, current_atr, current_price
    
    def generate_signal(self, latest_window: pd.DataFrame, 
                       tp_multiplier: float = 1.5, 
                       sl_multiplier: float = 1.0,
                       min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Generate trading signal from latest window
        
        Args:
            latest_window: DataFrame with latest OHLCV data
            tp_multiplier: TP multiplier for ATR
            sl_multiplier: SL multiplier for ATR
            min_confidence: Minimum confidence threshold
            
        Returns:
            Signal dictionary
        """
        try:
            # Prepare data
            sequence, current_atr, current_price = self.prepare_latest_window(latest_window)
            
            # Make prediction
            prediction = self.model.predict(sequence)
            probabilities = self.model.predict_proba(sequence)
            
            # Get confidence (probability of Long signal)
            confidence = probabilities[0][1]  # Probability of class 1 (Long)
            
            # Determine signal
            if confidence >= min_confidence and prediction[0] == 1:
                signal = "Long"
                tp = current_price + (tp_multiplier * current_atr)
                sl = current_price - (sl_multiplier * current_atr)
            elif confidence >= min_confidence and prediction[0] == 0:
                signal = "Short"
                tp = current_price - (tp_multiplier * current_atr)
                sl = current_price + (sl_multiplier * current_atr)
            else:
                signal = "None"
                tp = None
                sl = None
            
            result = {
                "signal": signal,
                "entry": current_price,
                "tp": tp,
                "sl": sl,
                "confidence": confidence,
                "atr": current_atr,
                "timestamp": latest_window.index[-1]
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return {
                "signal": "None",
                "entry": None,
                "tp": None,
                "sl": None,
                "confidence": 0.0,
                "atr": None,
                "timestamp": None,
                "error": str(e)
            }
    
    def generate_batch_signals(self, df: pd.DataFrame, 
                              tp_multiplier: float = 1.5,
                              sl_multiplier: float = 1.0,
                              min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Generate signals for a batch of data
        
        Args:
            df: DataFrame with OHLCV data
            tp_multiplier: TP multiplier for ATR
            sl_multiplier: SL multiplier for ATR
            min_confidence: Minimum confidence threshold
            
        Returns:
            DataFrame with signals
        """
        signals = []
        
        for i in range(self.sequence_length, len(df)):
            window = df.iloc[:i+1]
            signal = self.generate_signal(window, tp_multiplier, sl_multiplier, min_confidence)
            signals.append(signal)
        
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('timestamp', inplace=True)
        
        return signals_df


def load_signal_generator(model_path: str = "./models/hybrid_model.h5") -> SignalGenerator:
    """
    Load signal generator with trained model
    
    Args:
        model_path: Path to trained model
        
    Returns:
        SignalGenerator instance
    """
    return SignalGenerator(model_path)


def generate_signal_from_data(df: pd.DataFrame, 
                             model_path: str = "./models/hybrid_model.h5",
                             tp_multiplier: float = 1.5,
                             sl_multiplier: float = 1.0,
                             min_confidence: float = 0.6) -> Dict[str, Any]:
    """
    Generate signal from DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        model_path: Path to trained model
        tp_multiplier: TP multiplier for ATR
        sl_multiplier: SL multiplier for ATR
        min_confidence: Minimum confidence threshold
        
    Returns:
        Signal dictionary
    """
    generator = load_signal_generator(model_path)
    return generator.generate_signal(df, tp_multiplier, sl_multiplier, min_confidence)


if __name__ == "__main__":
    # Test signal generation
    from merge_data import load_and_merge_2024_data
    from features import calculate_technical_indicators
    
    try:
        # Load data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        
        # Test signal generation
        print("Testing signal generation...")
        signal = generate_signal_from_data(df_features)
        
        print("Generated signal:")
        for key, value in signal.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
