"""
FastAPI server for real-time signal generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn

from .signal import SignalGenerator, generate_signal_from_data
from .merge_data import load_and_merge_2024_data
from .features import calculate_technical_indicators


app = FastAPI(
    title="Bitcoin Scalping Signal API",
    description="Real-time Bitcoin scalping signal generation using LSTM + XGBoost",
    version="1.0.0"
)

# Global signal generator
signal_generator = None


class SignalRequest(BaseModel):
    """Request model for signal generation"""
    tp_multiplier: float = 1.5
    sl_multiplier: float = 1.0
    min_confidence: float = 0.6


class SignalResponse(BaseModel):
    """Response model for signal generation"""
    signal: str
    entry: Optional[float]
    tp: Optional[float]
    sl: Optional[float]
    confidence: float
    atr: Optional[float]
    timestamp: Optional[str]
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize signal generator on startup"""
    global signal_generator
    try:
        signal_generator = SignalGenerator()
        print("Signal generator initialized successfully")
    except Exception as e:
        print(f"Error initializing signal generator: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bitcoin Scalping Signal API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": signal_generator is not None
    }


@app.post("/signal", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """
    Generate trading signal from latest data
    
    Args:
        request: Signal generation parameters
        
    Returns:
        Trading signal with entry, TP, SL, and confidence
    """
    if signal_generator is None:
        raise HTTPException(status_code=500, detail="Signal generator not initialized")
    
    try:
        # Load latest data
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        
        # Generate signal
        signal = signal_generator.generate_signal(
            df_features,
            tp_multiplier=request.tp_multiplier,
            sl_multiplier=request.sl_multiplier,
            min_confidence=request.min_confidence
        )
        
        return SignalResponse(**signal)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")


@app.get("/signal/latest", response_model=SignalResponse)
async def get_latest_signal():
    """
    Get latest signal with default parameters
    
    Returns:
        Latest trading signal
    """
    request = SignalRequest()
    return await generate_signal(request)


@app.get("/model/info")
async def get_model_info():
    """
    Get model information
    
    Returns:
        Model configuration and status
    """
    if signal_generator is None:
        raise HTTPException(status_code=500, detail="Signal generator not initialized")
    
    return {
        "model_type": "Hybrid LSTM + XGBoost",
        "sequence_length": signal_generator.sequence_length,
        "feature_columns": len(signal_generator.feature_columns) if signal_generator.feature_columns else 0,
        "model_path": signal_generator.model_path,
        "status": "loaded"
    }


@app.get("/data/stats")
async def get_data_stats():
    """
    Get data statistics
    
    Returns:
        Data statistics and information
    """
    try:
        df = load_and_merge_2024_data()
        
        return {
            "total_bars": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            "price_range": {
                "min": float(df['close'].min()),
                "max": float(df['close'].max()),
                "current": float(df['close'].iloc[-1])
            },
            "volume_stats": {
                "avg_volume": float(df['volume'].mean()),
                "max_volume": float(df['volume'].max())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data stats: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
