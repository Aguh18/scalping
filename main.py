"""
Main entry point for Bitcoin Scalping Model
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import main as train_main
from src.serve import app
import uvicorn


def train_model():
    """Train the hybrid LSTM + XGBoost model"""
    print("Starting model training...")
    train_main()


def serve_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    print(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def test_signal():
    """Test signal generation"""
    from src.signal import generate_signal_from_data
    from src.merge_data import load_and_merge_2024_data
    from src.features import calculate_technical_indicators
    
    print("Testing signal generation...")
    
    try:
        # Load data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        
        # Generate signal
        signal = generate_signal_from_data(df_features)
        
        print("Generated signal:")
        for key, value in signal.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error testing signal: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bitcoin Scalping Model")
    parser.add_argument(
        "command",
        choices=["train", "serve", "test"],
        help="Command to run: train (train model), serve (start API), test (test signal)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for API server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model()
    elif args.command == "serve":
        serve_api(args.host, args.port)
    elif args.command == "test":
        test_signal()


if __name__ == "__main__":
    main()
