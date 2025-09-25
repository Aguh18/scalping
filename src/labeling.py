"""
Labeling module for creating trading signals with TP/SL logic
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def calculate_tp_sl_labels(df: pd.DataFrame, horizon: int = 5, tp_multiplier: float = 1.5, sl_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Calculate TP/SL labels for each bar
    
    Args:
        df: DataFrame with OHLCV data and ATR
        horizon: Number of bars to look ahead
        tp_multiplier: TP multiplier for ATR
        sl_multiplier: SL multiplier for ATR
        
    Returns:
        DataFrame with labeling columns
    """
    df_labeled = df.copy()
    
    # Initialize label columns
    df_labeled['entry'] = df_labeled['close']
    df_labeled['tp_long'] = np.nan
    df_labeled['sl_long'] = np.nan
    df_labeled['tp_short'] = np.nan
    df_labeled['sl_short'] = np.nan
    df_labeled['signal'] = np.nan
    df_labeled['label'] = np.nan  # 1 for Long, 0 for Short
    df_labeled['confidence'] = np.nan
    
    print("Calculating TP/SL labels...")
    
    for i in range(len(df_labeled) - horizon):
        current_price = df_labeled.iloc[i]['close']
        current_atr = df_labeled.iloc[i]['atr_14']
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
            
        # Calculate TP and SL levels
        tp_long = current_price + (tp_multiplier * current_atr)
        sl_long = current_price - (sl_multiplier * current_atr)
        tp_short = current_price - (tp_multiplier * current_atr)
        sl_short = current_price + (sl_multiplier * current_atr)
        
        # Look ahead to see which level is hit first
        future_data = df_labeled.iloc[i+1:i+1+horizon]
        
        if len(future_data) < horizon:
            continue
            
        # Check Long position
        long_tp_hit = (future_data['high'] >= tp_long).any()
        long_sl_hit = (future_data['low'] <= sl_long).any()
        
        # Check Short position
        short_tp_hit = (future_data['low'] <= tp_short).any()
        short_sl_hit = (future_data['high'] >= sl_short).any()
        
        # Determine signal based on which TP/SL is hit first
        long_signal = None
        short_signal = None
        
        # Check Long position
        if long_tp_hit and long_sl_hit:
            # Both hit, check which comes first
            tp_bar = future_data[future_data['high'] >= tp_long].index[0]
            sl_bar = future_data[future_data['low'] <= sl_long].index[0]
            long_signal = 1 if tp_bar <= sl_bar else 0
        elif long_tp_hit:
            long_signal = 1  # TP hit first
        elif long_sl_hit:
            long_signal = 0  # SL hit first
            
        # Check Short position
        if short_tp_hit and short_sl_hit:
            # Both hit, check which comes first
            tp_bar = future_data[future_data['low'] <= tp_short].index[0]
            sl_bar = future_data[future_data['high'] >= sl_short].index[0]
            short_signal = 1 if tp_bar <= sl_bar else 0
        elif short_tp_hit:
            short_signal = 1  # TP hit first
        elif short_sl_hit:
            short_signal = 0  # SL hit first
        
        # Store results
        df_labeled.iloc[i, df_labeled.columns.get_loc('tp_long')] = tp_long
        df_labeled.iloc[i, df_labeled.columns.get_loc('sl_long')] = sl_long
        df_labeled.iloc[i, df_labeled.columns.get_loc('tp_short')] = tp_short
        df_labeled.iloc[i, df_labeled.columns.get_loc('sl_short')] = sl_short
        
        # Determine final signal
        final_signal = None
        final_label = None
        confidence = 0.0
        
        if long_signal is not None and short_signal is not None:
            # Both Long and Short have valid signals
            if long_signal == 1 and short_signal == 0:
                # Long TP hit, Short SL hit - prefer Long
                final_signal = 'Long'
                final_label = 1
                confidence = 0.8
            elif long_signal == 0 and short_signal == 1:
                # Long SL hit, Short TP hit - prefer Short
                final_signal = 'Short'
                final_label = 1
                confidence = 0.8
            elif long_signal == 1 and short_signal == 1:
                # Both TP hit - prefer Long
                final_signal = 'Long'
                final_label = 1
                confidence = 0.7
            elif long_signal == 0 and short_signal == 0:
                # Both SL hit - prefer Long
                final_signal = 'Long'
                final_label = 0
                confidence = 0.3
        elif long_signal is not None:
            # Only Long signal available
            if long_signal == 1:
                final_signal = 'Long'
                final_label = 1
                confidence = 0.9
            else:
                final_signal = 'Long'
                final_label = 0
                confidence = 0.4
        elif short_signal is not None:
            # Only Short signal available
            if short_signal == 1:
                final_signal = 'Short'
                final_label = 1
                confidence = 0.9
            else:
                final_signal = 'Short'
                final_label = 0
                confidence = 0.4
        else:
            # No TP or SL hit within horizon - No trade
            final_signal = 'None'
            final_label = 0
            confidence = 0.0
        
        # Store final results
        df_labeled.iloc[i, df_labeled.columns.get_loc('signal')] = final_signal
        df_labeled.iloc[i, df_labeled.columns.get_loc('label')] = final_label
        df_labeled.iloc[i, df_labeled.columns.get_loc('confidence')] = confidence
    
    # Remove rows without labels
    df_labeled = df_labeled.dropna(subset=['signal'])
    
    print(f"Generated {len(df_labeled)} labeled samples")
    print(f"Signal distribution:")
    print(df_labeled['signal'].value_counts())
    
    return df_labeled


def create_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final trading signals with entry, TP, SL
    
    Args:
        df: DataFrame with labels
        
    Returns:
        DataFrame with trading signals
    """
    signals_df = df[['entry', 'tp_long', 'sl_long', 'tp_short', 'sl_short', 'signal', 'label', 'confidence']].copy()
    
    # Create final TP and SL columns
    signals_df['tp'] = np.where(
        signals_df['signal'] == 'Long',
        signals_df['tp_long'],
        np.where(
            signals_df['signal'] == 'Short',
            signals_df['tp_short'],
            np.nan  # No trade
        )
    )
    
    signals_df['sl'] = np.where(
        signals_df['signal'] == 'Long',
        signals_df['sl_long'],
        np.where(
            signals_df['signal'] == 'Short',
            signals_df['sl_short'],
            np.nan  # No trade
        )
    )
    
    # Select final columns
    final_signals = signals_df[['entry', 'tp', 'sl', 'signal', 'label', 'confidence']].copy()
    
    # Add risk-reward ratio (only for valid trades)
    valid_trades = final_signals['signal'].isin(['Long', 'Short'])
    final_signals['risk_reward'] = np.nan
    
    long_mask = final_signals['signal'] == 'Long'
    short_mask = final_signals['signal'] == 'Short'
    
    final_signals.loc[long_mask, 'risk_reward'] = (
        final_signals.loc[long_mask, 'tp'] - final_signals.loc[long_mask, 'entry']
    ) / (
        final_signals.loc[long_mask, 'entry'] - final_signals.loc[long_mask, 'sl']
    )
    
    final_signals.loc[short_mask, 'risk_reward'] = (
        final_signals.loc[short_mask, 'entry'] - final_signals.loc[short_mask, 'tp']
    ) / (
        final_signals.loc[short_mask, 'sl'] - final_signals.loc[short_mask, 'entry']
    )
    
    return final_signals


def validate_labels(df: pd.DataFrame) -> bool:
    """
    Validate the generated labels
    
    Args:
        df: DataFrame with labels
        
    Returns:
        True if valid, False otherwise
    """
    # Check for required columns
    required_cols = ['entry', 'tp', 'sl', 'signal', 'label', 'confidence']
    if not all(col in df.columns for col in required_cols):
        print("Missing required columns")
        return False
    
    # Check signal values
    valid_signals = ['Long', 'Short', 'None']
    if not df['signal'].isin(valid_signals).all():
        print("Invalid signal values")
        return False
    
    # Check label values
    if not df['label'].isin([0, 1]).all():
        print("Invalid label values")
        return False
    
    # Check TP/SL logic
    long_mask = df['signal'] == 'Long'
    short_mask = df['signal'] == 'Short'
    
    # For Long: TP > entry > SL
    if not (df.loc[long_mask, 'tp'] > df.loc[long_mask, 'entry']).all():
        print("Invalid Long TP/SL logic")
        return False
    
    if not (df.loc[long_mask, 'entry'] > df.loc[long_mask, 'sl']).all():
        print("Invalid Long TP/SL logic")
        return False
    
    # For Short: SL > entry > TP
    if not (df.loc[short_mask, 'sl'] > df.loc[short_mask, 'entry']).all():
        print("Invalid Short TP/SL logic")
        return False
    
    if not (df.loc[short_mask, 'entry'] > df.loc[short_mask, 'tp']).all():
        print("Invalid Short TP/SL logic")
        return False
    
    # Check confidence range
    if not ((df['confidence'] >= 0) & (df['confidence'] <= 1)).all():
        print("Invalid confidence values")
        return False
    
    print("Label validation passed")
    return True


if __name__ == "__main__":
    # Test the labeling
    from merge_data import load_and_merge_2024_data
    from features import calculate_technical_indicators
    
    try:
        # Load and prepare data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        
        # Calculate labels
        df_labeled = calculate_tp_sl_labels(df_features)
        
        # Create trading signals
        signals = create_trading_signals(df_labeled)
        
        # Validate labels
        validate_labels(signals)
        
        print("\nFirst 10 signals:")
        print(signals.head(10))
        
        print(f"\nSignal statistics:")
        print(f"Total signals: {len(signals)}")
        print(f"Long signals: {len(signals[signals['signal'] == 'Long'])}")
        print(f"Short signals: {len(signals[signals['signal'] == 'Short'])}")
        print(f"Average confidence: {signals['confidence'].mean():.3f}")
        print(f"Average risk-reward: {signals['risk_reward'].mean():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
