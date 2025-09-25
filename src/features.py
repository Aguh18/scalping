"""
Technical indicators and feature engineering module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import ta


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for the dataset
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    df_features = df.copy()
    
    # Price-based indicators
    df_features['return'] = df_features['close'].pct_change()
    df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
    
    # Moving averages
    df_features['sma_10'] = ta.trend.sma_indicator(df_features['close'], window=10)
    df_features['sma_30'] = ta.trend.sma_indicator(df_features['close'], window=30)
    df_features['ema_10'] = ta.trend.ema_indicator(df_features['close'], window=10)
    
    # RSI
    df_features['rsi_14'] = ta.momentum.rsi(df_features['close'], window=14)
    
    # ATR
    df_features['atr_14'] = ta.volatility.average_true_range(
        df_features['high'], df_features['low'], df_features['close'], window=14
    )
    
    # Volume indicators
    df_features['volume_sma'] = df_features['volume'].rolling(window=10).mean()
    df_features['volume_delta'] = df_features['volume'] - df_features['volume'].shift(1)
    df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
    
    # Volatility
    df_features['rolling_volatility'] = df_features['return'].rolling(window=20).std()
    df_features['price_volatility'] = (df_features['high'] - df_features['low']) / df_features['close']
    
    # Price position indicators
    df_features['price_position'] = (df_features['close'] - df_features['low']) / (df_features['high'] - df_features['low'])
    df_features['close_to_sma10'] = df_features['close'] / df_features['sma_10'] - 1
    df_features['close_to_sma30'] = df_features['close'] / df_features['sma_30'] - 1
    df_features['sma_ratio'] = df_features['sma_10'] / df_features['sma_30'] - 1
    
    # Momentum indicators
    df_features['momentum_5'] = df_features['close'] / df_features['close'].shift(5) - 1
    df_features['momentum_10'] = df_features['close'] / df_features['close'].shift(10) - 1
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df_features['close'], window=20, window_dev=2)
    df_features['bb_upper'] = bb.bollinger_hband()
    df_features['bb_lower'] = bb.bollinger_lband()
    df_features['bb_middle'] = bb.bollinger_mavg()
    df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
    df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
    
    # MACD
    macd = ta.trend.MACD(df_features['close'])
    df_features['macd'] = macd.macd()
    df_features['macd_signal'] = macd.macd_signal()
    df_features['macd_histogram'] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close'])
    df_features['stoch_k'] = stoch.stoch()
    df_features['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df_features['williams_r'] = ta.momentum.williams_r(df_features['high'], df_features['low'], df_features['close'])
    
    # Commodity Channel Index
    df_features['cci'] = ta.trend.cci(df_features['high'], df_features['low'], df_features['close'])
    
    # Average Directional Index
    df_features['adx'] = ta.trend.adx(df_features['high'], df_features['low'], df_features['close'])
    
    # Parabolic SAR
    df_features['psar'] = ta.trend.psar_up(df_features['high'], df_features['low'], df_features['close'])
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(df_features['high'], df_features['low'])
    df_features['ichimoku_a'] = ichimoku.ichimoku_a()
    df_features['ichimoku_b'] = ichimoku.ichimoku_b()
    df_features['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df_features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    
    # Price action features
    df_features['body_size'] = abs(df_features['close'] - df_features['open']) / df_features['close']
    df_features['upper_shadow'] = (df_features['high'] - np.maximum(df_features['open'], df_features['close'])) / df_features['close']
    df_features['lower_shadow'] = (np.minimum(df_features['open'], df_features['close']) - df_features['low']) / df_features['close']
    
    # Gap features
    df_features['gap'] = (df_features['open'] - df_features['close'].shift(1)) / df_features['close'].shift(1)
    
    print(f"Added {len(df_features.columns) - len(df.columns)} technical indicators")
    return df_features


def prepare_features_for_model(df: pd.DataFrame, feature_columns: Optional[list] = None) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for model training
    
    Args:
        df: DataFrame with technical indicators
        feature_columns: List of columns to use as features (if None, auto-select)
        
    Returns:
        Tuple of (feature_df, feature_columns)
    """
    if feature_columns is None:
        # Auto-select feature columns (exclude OHLCV and date)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Select features
    feature_df = df[feature_columns].copy()
    
    # Remove rows with NaN values
    initial_rows = len(feature_df)
    feature_df = feature_df.dropna()
    final_rows = len(feature_df)
    
    if initial_rows != final_rows:
        print(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    print(f"Selected {len(feature_columns)} features for model training")
    print(f"Feature columns: {feature_columns}")
    
    return feature_df, feature_columns


def normalize_features(df: pd.DataFrame, scaler: Optional[MinMaxScaler] = None, fit: bool = True) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler
    
    Args:
        df: DataFrame with features
        scaler: Pre-fitted scaler (if None, create new one)
        fit: Whether to fit the scaler
        
    Returns:
        Tuple of (normalized_df, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
    
    if fit:
        normalized_data = scaler.fit_transform(df.values)
    else:
        normalized_data = scaler.transform(df.values)
    
    normalized_df = pd.DataFrame(
        normalized_data,
        index=df.index,
        columns=df.columns
    )
    
    return normalized_df, scaler


def create_sequences(data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training
    
    Args:
        data: Normalized feature data
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test the feature calculation
    from merge_data import load_and_merge_2024_data
    
    try:
        # Load data
        df = load_and_merge_2024_data()
        print(f"Original data shape: {df.shape}")
        
        # Calculate technical indicators
        df_features = calculate_technical_indicators(df)
        print(f"Data with features shape: {df_features.shape}")
        
        # Prepare features
        feature_df, feature_cols = prepare_features_for_model(df_features)
        print(f"Feature data shape: {feature_df.shape}")
        
        # Normalize features
        normalized_df, scaler = normalize_features(feature_df)
        print(f"Normalized data shape: {normalized_df.shape}")
        
        # Create sequences
        X, y = create_sequences(normalized_df.values, sequence_length=60)
        print(f"Sequence data shapes - X: {X.shape}, y: {y.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
