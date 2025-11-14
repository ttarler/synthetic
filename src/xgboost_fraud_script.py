#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

def model_fn(model_dir):
    """Load model for SageMaker inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def train():
    """Train XGBoost model for fraud detection"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--num_round', type=int, default=100)
    
    args = parser.parse_args()
    
    # Load training data
    train_data = pd.read_csv(os.path.join(args.train, 'labeled_data.csv'), header=None)
    
    # Separate features and labels
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Set parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective,
        'eval_metric': 'auc'
    }
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round
    )
    
    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    print("Model training completed and saved")

if __name__ == '__main__':
    train()