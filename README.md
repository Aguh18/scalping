# Bitcoin Scalping Model

Model hybrid LSTM + XGBoost untuk scalping Bitcoin (BTCUSDT 5m) menggunakan data tahun 2024.

## 🚀 Quick Start di Google Colab

### 1. Clone Repository
```python
!git clone https://github.com/username/bitcoin-scalping-model.git
%cd bitcoin-scalping-model
```

### 2. Setup Environment
```python
!python colab_setup.py
```

### 3. Download Data
```python
!python download_data.py
```

### 4. Train Model
```python
!python main.py train
```

## 📋 Features

- **Model Hybrid**: LSTM sebagai encoder sequence + XGBoost sebagai classifier
- **Labeling Cerdas**: TP/SL dengan horizon 5 bar ke depan
- **40+ Indikator Teknikal**: RSI, ATR, Bollinger Bands, MACD, dll
- **Real-time Signal Generation**: API untuk generate signal
- **Signal Lengkap**: Entry price, TP, SL, confidence score

## 🏗️ Architecture

### Data Processing
- Load semua file BTCUSDT-5m-2024-*.csv
- Hitung 40+ indikator teknikal
- Normalisasi dengan MinMaxScaler

### Labeling
- Horizon: 5 bar ke depan
- Entry: harga close saat ini
- SL: 1 × ATR(14)
- TP: 1.5 × ATR(14)
- Signal: Long/Short/None berdasarkan TP/SL yang tercapai lebih dulu

### Model Hybrid
- **LSTM Encoder**: 2 layer (hidden=128, dropout=0.2)
- **XGBoost Classifier**: n_estimators=200, max_depth=8
- **Training**: Time-series split (70% train, 15% val, 15% test)

## 📁 Project Structure

```
bitcoin-scalping-model/
├── data/                          # CSV files (download with download_data.py)
├── src/                          # Source code
│   ├── merge_data.py            # Gabung CSV files
│   ├── features.py              # Indikator teknikal
│   ├── labeling.py              # TP/SL labeling
│   ├── model.py                 # Hybrid LSTM+XGBoost
│   ├── train.py                 # Training pipeline
│   ├── signal.py                # Signal generation
│   └── serve.py                 # FastAPI server
├── models/                       # Trained models (auto-created)
├── main.py                      # Entry point
├── colab_setup.py               # Colab setup script
├── download_data.py             # Data download script
├── colab_notebook.ipynb         # Jupyter notebook
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## 🎯 Usage

### Training
```bash
python main.py train
```

### API Server
```bash
python main.py serve
```

### Test Signal
```bash
python main.py test
```

## 📊 API Endpoints

Setelah menjalankan `python main.py serve`:

### Generate Signal
```bash
curl -X POST "http://localhost:8000/signal" \
     -H "Content-Type: application/json" \
     -d '{
       "tp_multiplier": 1.5,
       "sl_multiplier": 1.0,
       "min_confidence": 0.6
     }'
```

### Response Format
```json
{
  "signal": "Long",
  "entry": 45210.50,
  "tp": 45840.25,
  "sl": 44980.20,
  "confidence": 0.73,
  "atr": 420.15,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.15+
- XGBoost 2.0+
- Pandas, NumPy, Scikit-learn
- FastAPI, Uvicorn

## 📈 Performance

- **Data**: 105,408 bar BTCUSDT 5m 2024
- **Training Time**: 30-60 menit (Colab GPU)
- **Model Size**: ~50MB
- **Signal Generation**: <1 detik

## 🎉 Results

Model menghasilkan signal trading dengan:
- Entry price yang akurat
- TP/SL berdasarkan ATR
- Confidence score untuk filtering
- Real-time API untuk trading

## 📝 License

MIT License

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

Jika ada pertanyaan atau masalah, silakan buat issue di GitHub repository.

## ⚡ Auto Run di Google Colab

### **Opsi 1: Single Cell Auto-Run (Paling Mudah)**

Copy-paste code ini ke **1 cell** di Colab:

```python
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
```

### **Opsi 2: Notebook Auto-Run**

1. Upload `colab_auto_run.ipynb` ke Colab
2. Run semua cell secara berurutan
3. Model akan otomatis trained dan results downloaded

### **Opsi 3: Manual Step-by-Step**

```python
# 1. Clone repository
!git clone https://github.com/username/bitcoin-scalping-model.git
%cd bitcoin-scalping-model

# 2. Setup environment
!python colab_setup.py

# 3. Download data
!python download_data.py

# 4. Train model
!python main.py train

# 5. Test signal
!python main.py test
```

## 🎯 **Expected Results**

Setelah auto-run selesai, Anda akan mendapatkan:

- ✅ **Trained Model** dengan performance metrics
- ✅ **Signal Generator** untuk real-time trading
- ✅ **Results Package** yang bisa didownload
- ✅ **API Server** untuk deployment

**Total waktu: 30-60 menit (tergantung GPU Colab)**
