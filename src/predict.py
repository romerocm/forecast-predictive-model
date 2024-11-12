import os
import numpy as np
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model import SalesPredictor

def generate_predictions():
    # Load data
    _, test_df = load_data()
    
    # Preprocess test data
    X_test = preprocess_data(test_df, is_training=False)
    
    # Load model and make predictions
    predictor = SalesPredictor()
    predictor.load_model()
    predictions = predictor.predict(X_test)
    
    # Round predictions to integers
    predictions = np.round(predictions).astype(int)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'num_sold': predictions
    })
    
    # Ensure the output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save predictions in the required format
    submission.to_csv('data/processed/submission.csv', index=False)
    print("\nPredictions saved to data/processed/submission.csv")
    print(f"Generated {len(predictions)} predictions")
    print("\nFirst few predictions:")
    print(submission.head().to_string())

if __name__ == "__main__":
    generate_predictions()
