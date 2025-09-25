"""
LSTM model architecture for Bitcoin scalping
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import Tuple, Optional, Dict, Any
import joblib
import os


class HybridLSTMXGBModel:
    """
    Hybrid LSTM + XGBoost model for Bitcoin scalping
    LSTM acts as sequence encoder, XGBoost as final classifier
    """
    
    def __init__(self, sequence_length: int = 60, n_features: int = 50, 
                 hidden_units: int = 128, dropout_rate: float = 0.2):
        """
        Initialize the hybrid model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            hidden_units: Number of LSTM hidden units
            dropout_rate: Dropout rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        self.lstm_encoder = None
        self.xgb_classifier = None
        self.scaler = None
        self.feature_columns = None
        
    def build_lstm_encoder(self) -> Model:
        """
        Build LSTM encoder for sequence embedding
        
        Returns:
            LSTM encoder model
        """
        # Input layer
        sequence_input = Input(shape=(self.sequence_length, self.n_features), name='sequence_input')
        
        # First LSTM layer
        lstm1 = LSTM(
            self.hidden_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='lstm1'
        )(sequence_input)
        
        # Second LSTM layer
        lstm2 = LSTM(
            self.hidden_units,
            return_sequences=False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='lstm2'
        )(lstm1)
        
        # Dense layer for embedding
        embedding = Dense(
            64, 
            activation='relu', 
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            name='embedding'
        )(lstm2)
        
        embedding = Dropout(self.dropout_rate)(embedding)
        
        # Create encoder model
        encoder = Model(inputs=sequence_input, outputs=embedding, name='lstm_encoder')
        
        return encoder
    
    def build_hybrid_model(self, static_features_dim: int = 0) -> Tuple[Model, Any]:
        """
        Build hybrid LSTM + XGBoost model
        
        Args:
            static_features_dim: Dimension of static technical features
            
        Returns:
            Tuple of (LSTM encoder, XGBoost classifier)
        """
        # Build LSTM encoder
        lstm_encoder = self.build_lstm_encoder()
        
        # XGBoost classifier
        xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        return lstm_encoder, xgb_classifier
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              X_static_train: Optional[np.ndarray] = None,
              X_static_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 128,
              model_path: str = "./models/hybrid_model.h5") -> Dict[str, Any]:
        """
        Train the hybrid model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            X_static_train: Static technical features for training
            X_static_val: Static technical features for validation
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save the model
            
        Returns:
            Training history
        """
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        return self._train_hybrid(X_train, y_train, X_val, y_val, 
                                 X_static_train, X_static_val, 
                                 epochs, batch_size, model_path)
    
    def _train_hybrid(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_static_train: Optional[np.ndarray],
                     X_static_val: Optional[np.ndarray],
                     epochs: int, batch_size: int, model_path: str) -> Dict[str, Any]:
        """Train hybrid LSTM + XGBoost model"""
        
        # Build models
        lstm_encoder, xgb_classifier = self.build_hybrid_model()
        
        # Step 1: Train LSTM encoder with auxiliary task (binary classification)
        print("Step 1: Training LSTM encoder...")
        
        # Add classification head to LSTM for training
        lstm_input = Input(shape=(self.sequence_length, self.n_features))
        lstm_embedding = lstm_encoder(lstm_input)
        classification_output = Dense(1, activation='sigmoid', name='classification')(lstm_embedding)
        
        # Create training model
        training_model = Model(inputs=lstm_input, outputs=classification_output)
        training_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for LSTM training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_path.replace('.h5', '_lstm.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train LSTM encoder
        lstm_history = training_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Step 2: Extract embeddings using trained LSTM
        print("Step 2: Extracting LSTM embeddings...")
        X_train_embeddings = lstm_encoder.predict(X_train, batch_size=batch_size)
        X_val_embeddings = lstm_encoder.predict(X_val, batch_size=batch_size)
        
        # Step 3: Combine embeddings with static features if provided
        if X_static_train is not None and X_static_val is not None:
            print("Step 3: Combining embeddings with static features...")
            X_train_combined = np.concatenate([X_train_embeddings, X_static_train], axis=1)
            X_val_combined = np.concatenate([X_val_embeddings, X_static_val], axis=1)
        else:
            X_train_combined = X_train_embeddings
            X_val_combined = X_val_embeddings
        
        # Step 4: Train XGBoost classifier
        print("Step 4: Training XGBoost classifier...")
        xgb_classifier.fit(
            X_train_combined, y_train,
            eval_set=[(X_val_combined, y_val)],
            early_stopping_rounds=20,
            verbose=1
        )
        
        # Save models
        lstm_encoder.save(model_path.replace('.h5', '_encoder.h5'))
        joblib.dump(xgb_classifier, model_path.replace('.h5', '_xgb.pkl'))
        
        # Store models
        self.lstm_encoder = lstm_encoder
        self.xgb_classifier = xgb_classifier
        
        print("Hybrid model training completed!")
        return lstm_history.history
    
    def predict(self, X: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using hybrid model
        
        Args:
            X: Input sequences
            X_static: Static technical features
            
        Returns:
            Predictions (0: No trade/Short, 1: Long)
        """
        # Extract embeddings using LSTM encoder
        embeddings = self.lstm_encoder.predict(X)
        
        # Combine with static features if provided
        if X_static is not None:
            combined_features = np.concatenate([embeddings, X_static], axis=1)
        else:
            combined_features = embeddings
        
        # Get predictions from XGBoost
        predictions = self.xgb_classifier.predict(combined_features)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray, X_static: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get prediction probabilities using hybrid model
        
        Args:
            X: Input sequences
            X_static: Static technical features
            
        Returns:
            Prediction probabilities [No trade/Short, Long]
        """
        # Extract embeddings using LSTM encoder
        embeddings = self.lstm_encoder.predict(X)
        
        # Combine with static features if provided
        if X_static is not None:
            combined_features = np.concatenate([embeddings, X_static], axis=1)
        else:
            combined_features = embeddings
        
        # Get probabilities from XGBoost
        probabilities = self.xgb_classifier.predict_proba(combined_features)
        
        return probabilities
    
    def save_model(self, model_path: str):
        """Save the trained hybrid model"""
        self.lstm_encoder.save(model_path.replace('.h5', '_encoder.h5'))
        joblib.dump(self.xgb_classifier, model_path.replace('.h5', '_xgb.pkl'))
        print(f"Hybrid model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained hybrid model"""
        self.lstm_encoder = tf.keras.models.load_model(model_path.replace('.h5', '_encoder.h5'))
        self.xgb_classifier = joblib.load(model_path.replace('.h5', '_xgb.pkl'))
        print(f"Hybrid model loaded from {model_path}")


def create_sequences_with_labels(data: np.ndarray, labels: np.ndarray, 
                                sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with corresponding labels
    
    Args:
        data: Feature data
        labels: Target labels
        sequence_length: Length of sequences
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)


def split_time_series_data(X: np.ndarray, y: np.ndarray, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    Split time series data maintaining temporal order
    
    Args:
        X: Feature sequences
        y: Labels
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test the model
    from merge_data import load_and_merge_2024_data
    from features import calculate_technical_indicators, prepare_features_for_model, normalize_features
    from labeling import calculate_tp_sl_labels, create_trading_signals
    
    try:
        # Load and prepare data
        print("Loading data...")
        df = load_and_merge_2024_data()
        df_features = calculate_technical_indicators(df)
        
        # Create labels
        print("Creating labels...")
        df_labeled = calculate_tp_sl_labels(df_features)
        signals = create_trading_signals(df_labeled)
        
        # Prepare features
        print("Preparing features...")
        feature_df, feature_cols = prepare_features_for_model(df_features, feature_columns=None)
        normalized_df, scaler = normalize_features(feature_df)
        
        # Align data
        common_index = normalized_df.index.intersection(signals.index)
        X_data = normalized_df.loc[common_index].values
        y_data = signals.loc[common_index]['label'].values
        
        # Create sequences
        print("Creating sequences...")
        X, y = create_sequences_with_labels(X_data, y_data, sequence_length=60)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_data(X, y)
        
        print(f"Data shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Test model creation
        print("Testing hybrid model creation...")
        model = HybridLSTMXGBModel(
            sequence_length=60,
            n_features=X_train.shape[2]
        )
        
        lstm_encoder, xgb_classifier = model.build_hybrid_model()
        print(f"LSTM encoder created with {lstm_encoder.count_params()} parameters")
        print(f"XGBoost classifier created")
        
    except Exception as e:
        print(f"Error: {e}")
