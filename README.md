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
