import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import logging

# Configure logging
logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        
    Returns:
        dict: Model performance metrics
    """
    try:
        logger.info("Training Random Forest model")
        
        # Create and train model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Get feature importances
        feature_importances = rf.feature_importances_
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        
        logger.info(f"Random Forest model trained: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
        
        return {
            'model': rf,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'feature_importances': feature_importances,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}")
        raise

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an XGBoost model.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        
    Returns:
        dict: Model performance metrics
    """
    try:
        logger.info("Training XGBoost model")
        
        # Create and train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Get feature importances
        feature_importances = xgb_model.feature_importances_
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
        
        logger.info(f"XGBoost model trained: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
        
        return {
            'model': xgb_model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'feature_importances': feature_importances,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        raise

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Gradient Boosting model.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        
    Returns:
        dict: Model performance metrics
    """
    try:
        logger.info("Training Gradient Boosting model")
        
        # Create and train model
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        # Make predictions
        y_pred = gb.predict(X_test)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Get feature importances
        feature_importances = gb.feature_importances_
        
        # Cross-validation
        cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')
        
        logger.info(f"Gradient Boosting model trained: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
        
        return {
            'model': gb,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'feature_importances': feature_importances,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error training Gradient Boosting model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, model_type='random_forest'):
    """
    Perform hyperparameter optimization for a specified model.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        model_type (str, optional): Type of model to optimize. 
                                   Options: 'random_forest', 'xgboost', 'gradient_boosting'.
                                   Default is 'random_forest'.
        
    Returns:
        dict: Best parameters found
    """
    try:
        logger.info(f"Optimizing hyperparameters for {model_type}")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
            
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {}
        
        # Use GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
    except Exception as e:
        logger.error(f"Error optimizing hyperparameters: {str(e)}")
        raise

def train_evaluate_ml_models(X_train, y_train, X_test, y_test, models=None):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target vector
        models (list, optional): List of models to train. 
                               Options: 'random_forest', 'xgboost', 'gradient_boosting'.
                               Default is all models.
        
    Returns:
        tuple: (results_df, best_model, feature_importances)
    """
    try:
        if models is None:
            models = ['random_forest', 'xgboost', 'gradient_boosting']
            
        results = []
        feature_importances = {}
        best_model = None
        best_f1 = -1
        
        # Train requested models
        for model_type in models:
            if model_type == 'random_forest':
                result = train_random_forest(X_train, y_train, X_test, y_test)
                
            elif model_type == 'xgboost':
                result = train_xgboost(X_train, y_train, X_test, y_test)
                
            elif model_type == 'gradient_boosting':
                result = train_gradient_boosting(X_train, y_train, X_test, y_test)
                
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            # Store results
            results.append({
                'model': model_type,
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'cv_mean': np.mean(result['cv_scores']),
                'cv_std': np.std(result['cv_scores'])
            })
            
            # Store feature importances
            feature_importances[model_type] = result['feature_importances']
            
            # Keep track of best model
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_model = result['model']
                best_predictions = result['predictions']
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Add best predictions to results for backtesting
        results_df['best_predictions'] = best_predictions if 'best_predictions' in locals() else None
        
        return results_df, best_model, feature_importances
        
    except Exception as e:
        logger.error(f"Error training and evaluating models: {str(e)}")
        raise
