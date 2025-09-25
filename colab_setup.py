"""
Google Colab Setup Script
Run this in Colab to setup the Bitcoin Scalping Model
"""

import os
import subprocess
import sys

def setup_colab():
    """Setup environment for Google Colab"""
    
    print("🚀 Setting up Bitcoin Scalping Model for Google Colab...")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "tensorflow",
        "xgboost",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pydantic",
        "ta",
        "requests"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Create directories
    print("📁 Creating directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run download_data.py to get BTCUSDT data")
    print("2. Run main.py train to start training")
    print("3. Or use the Jupyter notebook for step-by-step process")

def check_gpu():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        print(f"🔍 GPU Available: {tf.config.list_physical_devices('GPU')}")
        print(f"🔍 TensorFlow Version: {tf.__version__}")
        
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"❌ Error setting GPU memory growth: {e}")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

def download_sample_data():
    """Download sample data for testing"""
    print("📥 Downloading sample data...")
    try:
        from download_data import download_2024_data
        download_2024_data()
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("Please run download_data.py manually")

if __name__ == "__main__":
    setup_colab()
    check_gpu()
    print("\n🎯 Ready to start training!")
    print("Run: python main.py train")
