"""
Single cell untuk auto-run di Google Colab
Copy-paste code ini ke 1 cell di Colab untuk auto-run semua
"""

# Bitcoin Scalping Model - Auto Run (Single Cell)
print("🚀 Bitcoin Scalping Model - Auto Run")
print("=" * 50)

# Step 1: Clone repository
print("\n📥 Step 1: Cloning repository...")
!git clone https://github.com/username/bitcoin-scalping-model.git
%cd bitcoin-scalping-model

# Step 2: Setup environment
print("\n🔧 Step 2: Setting up environment...")
!python colab_setup.py

# Step 3: Download data
print("\n📊 Step 3: Downloading data...")
!python download_data.py

# Step 4: Train model
print("\n🏋️ Step 4: Training model...")
!python main.py train

# Step 5: Test signal generation
print("\n🎯 Step 5: Testing signal generation...")
!python main.py test

# Step 6: Show results
print("\n📈 Step 6: Showing results...")
import joblib
import os

if os.path.exists('models/evaluation_results.pkl'):
    results = joblib.load('models/evaluation_results.pkl')
    print("📊 Model Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")

# Step 7: Download results
print("\n📦 Step 7: Creating results package...")
import zipfile

with zipfile.ZipFile('bitcoin_scalping_results.zip', 'w') as zipf:
    for root, dirs, files in os.walk('models'):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, file)
    
    for root, dirs, files in os.walk('src'):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, file)
    
    main_files = ['main.py', 'requirements.txt', 'README.md']
    for file in main_files:
        if os.path.exists(file):
            zipf.write(file, file)

print("✅ Results package created: bitcoin_scalping_results.zip")

# Download results
from google.colab import files
files.download('bitcoin_scalping_results.zip')

print("\n🎉 Auto-run completed successfully!")
print("📊 Model is ready for trading!")
print("📥 Results downloaded to your computer!")
