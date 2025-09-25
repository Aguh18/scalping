"""
Script to download BTCUSDT 5m data for 2024
Run this in Google Colab to get the data
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def download_binance_data(symbol, interval, start_date, end_date):
    """
    Download data from Binance API
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            # Check if response is successful
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                break
            
            data = response.json()
            
            # Check if data is valid
            if not data or not isinstance(data, list):
                print(f"Invalid data format: {type(data)}")
                break
            
            # Check if data has the expected structure
            if len(data) > 0 and len(data[0]) != 12:
                print(f"Unexpected data structure: {len(data[0])} columns instead of 12")
                break
                
            all_data.extend(data)
            current_ts = data[-1][0] + 1  # Next timestamp
            
            print(f"Downloaded {len(data)} bars, total: {len(all_data)}")
            time.sleep(0.1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_data

def save_data_to_csv(data, filename):
    """
    Save data to CSV file
    """
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Select relevant columns and rename
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove NaN values
    df = df.dropna()
    
    # Save to CSV without index
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")

def check_existing_data():
    """
    Check if data already exists in ./data folder
    """
    import os
    import glob
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ Data directory not found!")
        return False
    
    # Check for existing CSV files
    pattern = "data/BTCUSDT-5m-2024-*.csv"
    csv_files = glob.glob(pattern)
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} existing CSV files:")
        for file in sorted(csv_files):
            print(f"  - {os.path.basename(file)}")
        return True
    else:
        print("❌ No existing CSV files found in ./data folder")
        print("Please upload your BTCUSDT 5m 2024 CSV files to ./data folder")
        return False

def download_2024_data():
    """
    Check for existing data or download if needed
    """
    print("🔍 Checking for existing data...")
    
    if check_existing_data():
        print("\n✅ Using existing data files!")
        return True
    else:
        print("\n📥 No existing data found. Downloading from Binance...")
        
        import os
        os.makedirs('data', exist_ok=True)
        
        # Download data for each month in 2024
        months = [
            ('2024-01-01', '2024-01-31'),
            ('2024-02-01', '2024-02-29'),
            ('2024-03-01', '2024-03-31'),
            ('2024-04-01', '2024-04-30'),
            ('2024-05-01', '2024-05-31'),
            ('2024-06-01', '2024-06-30'),
            ('2024-07-01', '2024-07-31'),
            ('2024-08-01', '2024-08-31'),
            ('2024-09-01', '2024-09-30'),
            ('2024-10-01', '2024-10-31'),
            ('2024-11-01', '2024-11-30'),
            ('2024-12-01', '2024-12-31')
        ]
        
        for i, (start_date, end_date) in enumerate(months, 1):
            print(f"\nDownloading {start_date} to {end_date}...")
            
            data = download_binance_data('BTCUSDT', '5m', start_date, end_date)
            
            if data:
                filename = f'data/BTCUSDT-5m-2024-{i:02d}.csv'
                save_data_to_csv(data, filename)
            else:
                print(f"No data for {start_date} to {end_date}")
        
        print("\n✅ All 2024 data downloaded!")
        return True

if __name__ == "__main__":
    download_2024_data()
