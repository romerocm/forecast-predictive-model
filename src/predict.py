from data_preprocessing import load_data, preprocess_data
from model import SalesPredictor
import pandas as pd

def generate_predictions():
    # Load data
    _, test_df = load_data()
    
    # Preprocess test data
    X_test = preprocess_data(test_df, is_training=False)
    
    # Load model and make predictions
    predictor = SalesPredictor()
    predictor.load_model()
    predictions = predictor.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'num_sold': predictions
    })
    submission.to_csv('data/processed/submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    generate_predictions()
