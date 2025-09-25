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
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_ts = data[-1][0] + 1  # Next timestamp
            
            print(f"Downloaded {len(data)} bars, total: {len(all_data)}")
            time.sleep(0.1)  # Rate limiting
            
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Select relevant columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")

def download_2024_data():
    """
    Download all 2024 data
    """
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

if __name__ == "__main__":
    download_2024_data()
