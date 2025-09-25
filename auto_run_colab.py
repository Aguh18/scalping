"""
Auto-run script for Google Colab
This script will automatically run all cells in the notebook
"""

import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and print the result"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Error!")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main auto-run function"""
    print("🎯 Bitcoin Scalping Model - Auto Run")
    print("=" * 50)
    
    # Step 1: Clone repository
    if not run_command("git clone https://github.com/username/bitcoin-scalping-model.git", 
                      "Cloning repository"):
        print("❌ Failed to clone repository. Please check the URL.")
        return
    
    # Step 2: Change directory
    if not run_command("cd bitcoin-scalping-model", "Changing to project directory"):
        print("❌ Failed to change directory.")
        return
    
    # Step 3: Setup environment
    if not run_command("python colab_setup.py", "Setting up environment"):
        print("❌ Failed to setup environment.")
        return
    
    # Step 4: Download data
    if not run_command("python download_data.py", "Downloading data"):
        print("❌ Failed to download data.")
        return
    
    # Step 5: Train model
    if not run_command("python main.py train", "Training model"):
        print("❌ Failed to train model.")
        return
    
    # Step 6: Test signal generation
    if not run_command("python main.py test", "Testing signal generation"):
        print("❌ Failed to test signal generation.")
        return
    
    print("\n🎉 Auto-run completed successfully!")
    print("📊 Model is ready for trading!")

if __name__ == "__main__":
    main()
