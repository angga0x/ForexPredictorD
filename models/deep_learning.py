import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_lstm_model(n_features, units=64, dropout_rate=0.2):
    """
    Create an LSTM model for time series classification.
    
    Args:
        n_features (int): Number of input features
        units (int, optional): Number of LSTM units. Default is 64.
        dropout_rate (float, optional): Dropout rate. Default is 0.2.
        
    Returns:
        tensorflow.keras.models.Sequential: LSTM model
    """
    try:
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(None, n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            LSTM(units=units//2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(units=32, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate/2),
            
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating LSTM model: {str(e)}")
        raise

def train_evaluate_lstm(X_train, y_train, X_test, y_test, n_features=None, epochs=50, batch_size=32):
    """
    Train and evaluate an LSTM model.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        n_features (int, optional): Number of input features. Default is None.
        epochs (int, optional): Number of training epochs. Default is 50.
        batch_size (int, optional): Batch size. Default is 32.
        
    Returns:
        dict: Model performance metrics
    """
    try:
        logger.info("Training LSTM model")
        
        # Prepare data for LSTM (reshape to 3D: [samples, time_steps, features])
        if len(X_train.shape) == 2:
            # If not already in sequence format, reshape to a single time step
            if n_features is None:
                n_features = X_train.shape[1]
            
            X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, n_features))
            X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, n_features))
        else:
            # If already in sequence format, use as is
            X_train_lstm = X_train
            X_test_lstm = X_test
            if n_features is None:
                n_features = X_train.shape[2]
        
        # Create model
        model = create_lstm_model(n_features)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Make predictions
        y_pred_prob = model.predict(X_test_lstm)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"LSTM model trained: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
        
        return {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'history': history,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        # Return a dictionary with error information
        return {
            'model': None,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'error': str(e)
        }

def create_rnn_model(n_features, units=64, dropout_rate=0.2):
    """
    Create a simple RNN model as an alternative to LSTM.
    
    Args:
        n_features (int): Number of input features
        units (int, optional): Number of RNN units. Default is 64.
        dropout_rate (float, optional): Dropout rate. Default is 0.2.
        
    Returns:
        tensorflow.keras.models.Sequential: RNN model
    """
    try:
        model = Sequential([
            tf.keras.layers.SimpleRNN(units=units, return_sequences=True, input_shape=(None, n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            tf.keras.layers.SimpleRNN(units=units//2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(units=32, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate/2),
            
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating RNN model: {str(e)}")
        raise

def create_cnn_model(n_features, filters=64, dropout_rate=0.2):
    """
    Create a 1D CNN model as an alternative to LSTM.
    
    Args:
        n_features (int): Number of input features
        filters (int, optional): Number of CNN filters. Default is 64.
        dropout_rate (float, optional): Dropout rate. Default is 0.2.
        
    Returns:
        tensorflow.keras.models.Sequential: CNN model
    """
    try:
        model = Sequential([
            tf.keras.layers.Conv1D(filters=filters, kernel_size=3, activation='relu', input_shape=(None, n_features)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            tf.keras.layers.Conv1D(filters=filters//2, kernel_size=3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(units=32, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate/2),
            
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating CNN model: {str(e)}")
        raise

def save_model(model, filepath):
    """
    Save a trained model to a file.
    
    Args:
        model: TensorFlow model to save
        filepath (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_model(filepath):
    """
    Load a trained model from a file.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        tensorflow.keras.models.Model: Loaded model or None if error occurs
    """
    try:
        model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None
