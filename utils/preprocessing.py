import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)

def prepare_data_for_training(df, prediction_horizon=5, train_size=0.8, target_column='close'):
    """
    Prepare data for model training by creating features and target variables.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        prediction_horizon (int, optional): Number of periods ahead to predict. Default is 5.
        train_size (float, optional): Proportion of data to use for training. Default is 0.8.
        target_column (str, optional): Column to use for target generation. Default is 'close'.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    try:
        logger.info(f"Preparing data for training with prediction horizon: {prediction_horizon}")
        
        # Create a copy of the dataframe
        data = df.copy()
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Create target variable (1 if price goes up after horizon, 0 otherwise)
        data['target'] = (data[target_column].shift(-prediction_horizon) > data[target_column]).astype(int)
        
        # Remove rows where target is NaN (at the end of the dataframe)
        data = data.dropna(subset=['target'])
        
        # Select features (exclude date, target, and raw OHLCV data)
        exclude_columns = ['date', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Filter out any remaining NaN values
        for col in feature_columns:
            if data[col].isna().any():
                logger.warning(f"Column {col} contains NaN values. Filling with mean.")
                data[col] = data[col].fillna(data[col].mean())
        
        # Create feature matrix and target vector
        X = data[feature_columns]
        y = data['target']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Split into training and test sets
        split_idx = int(len(X_scaled) * train_size)
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Data prepared: X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")
        raise

def feature_selection(X_train, y_train, X_test, method='correlation', n_features=10):
    """
    Select the most important features for model training.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        method (str, optional): Method to use for feature selection. 
                               Options: 'correlation', 'mutual_information', 'recursive_feature_elimination'.
                               Default is 'correlation'.
        n_features (int, optional): Number of features to select. Default is 10.
        
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_feature_names)
    """
    try:
        logger.info(f"Performing feature selection using method: {method}")
        
        if method == 'correlation':
            # Calculate correlation with target
            correlations = {}
            for feature in X_train.columns:
                correlation = abs(X_train[feature].corr(y_train))
                correlations[feature] = correlation
            
            # Sort by absolute correlation
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Select top n_features
            selected_features = [f[0] for f in sorted_features[:n_features]]
            
        elif method == 'mutual_information':
            # Use mutual information for feature selection
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(X_train, y_train)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = X_train.columns[selected_indices].tolist()
            
        elif method == 'recursive_feature_elimination':
            # Use RFE with Random Forest classifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            selector.fit(X_train, y_train)
            
            # Get selected feature indices
            selected_features = X_train.columns[selector.support_].tolist()
            
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using all features.")
            selected_features = X_train.columns.tolist()
        
        # Select features from train and test sets
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return X_train_selected, X_test_selected, selected_features
        
    except Exception as e:
        logger.error(f"Error during feature selection: {str(e)}")
        # Return original data if error occurs
        return X_train, X_test, X_train.columns.tolist()

def prepare_sequence_data(df, seq_length=10, prediction_horizon=5, train_size=0.8, target_column='close'):
    """
    Prepare sequential data for LSTM model.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        seq_length (int, optional): Length of input sequences. Default is 10.
        prediction_horizon (int, optional): Number of periods ahead to predict. Default is 5.
        train_size (float, optional): Proportion of data to use for training. Default is 0.8.
        target_column (str, optional): Column to use for target generation. Default is 'close'.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        logger.info(f"Preparing sequence data with sequence length: {seq_length}")
        
        # Create a copy of the dataframe
        data = df.copy()
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Create target variable (1 if price goes up after horizon, 0 otherwise)
        data['target'] = (data[target_column].shift(-prediction_horizon) > data[target_column]).astype(int)
        
        # Remove rows where target is NaN (at the end of the dataframe)
        data = data.dropna(subset=['target'])
        
        # Select features (exclude date, target, and raw OHLCV data)
        exclude_columns = ['date', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Filter out any remaining NaN values
        for col in feature_columns:
            if data[col].isna().any():
                logger.warning(f"Column {col} contains NaN values. Filling with mean.")
                data[col] = data[col].fillna(data[col].mean())
        
        # Create feature matrix and target vector
        X = data[feature_columns].values
        y = data['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - seq_length - prediction_horizon + 1):
            X_seq.append(X_scaled[i:(i + seq_length)])
            y_seq.append(y[i + seq_length + prediction_horizon - 1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split into training and test sets
        split_idx = int(len(X_seq) * train_size)
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        logger.info(f"Sequence data prepared: X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        logger.error(f"Error preparing sequence data: {str(e)}")
        raise
