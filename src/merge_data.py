"""
Data merging module for combining all 2024 BTCUSDT 5m CSV files
"""

import pandas as pd
import glob
import os
from typing import Optional


def load_and_merge_2024_data(data_dir: str = "./data") -> pd.DataFrame:
    """
    Load and merge all BTCUSDT 5m CSV files from 2024
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Combined DataFrame with proper datetime index
    """
    # Find all 2024 CSV files
    pattern = os.path.join(data_dir, "BTCUSDT-5m-2024-*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No 2024 CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV files for 2024")
    
    # Load and combine all files
    dataframes = []
    
    for file_path in sorted(csv_files):
        print(f"Loading {os.path.basename(file_path)}...")
        
        try:
            # Try to read with header first
            df = pd.read_csv(file_path)
            
            # Check if it has proper column names
            if 'open' in df.columns and 'high' in df.columns:
                # Already has proper column names
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    # Use first column as date
                    df['date'] = pd.to_datetime(df.iloc[:, 0])
            else:
                # No header, try to detect format
                df = pd.read_csv(file_path, header=None)
                
                # Check number of columns to determine format
                if len(df.columns) == 6:
                    # Simple format: date, open, high, low, close, volume
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    df['date'] = pd.to_datetime(df['date'])
                elif len(df.columns) == 12:
                    # Binance format
                    df.columns = [
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ]
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    # Try to use first 6 columns
                    df = df.iloc[:, :6]
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    df['date'] = pd.to_datetime(df['date'])
            
            # Select relevant columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna()
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by date and set as index
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    combined_df.set_index('date', inplace=True)
    
    # Remove duplicates if any
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the merged dataset
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Missing values found in dataset")
        return False
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        print("Warning: Non-positive prices found")
        return False
    
    # Check OHLC logic
    invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                   (df['low'] > df['open']) | (df['low'] > df['close'])
    
    if invalid_ohlc.any():
        print("Warning: Invalid OHLC data found")
        return False
    
    print("Data validation passed")
    return True


if __name__ == "__main__":
    # Test the data loading
    try:
        df = load_and_merge_2024_data()
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print(f"\nData types:\n{df.dtypes}")
        
        # Validate data
        validate_data(df)
        
        # Save combined dataset
        output_path = "./data/btcusdt_5m_2024_combined.csv"
        df.to_csv(output_path)
        print(f"\nCombined dataset saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
