from data_preprocessing import load_data, preprocess_data
from model import SalesPredictor
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    train_df, _ = load_data()
    
    print("Preprocessing data...")
    # Use only 2017-2021 data for training
    train_years = list(range(2017, 2022))
    X_train, y_train, countries = preprocess_data(train_df, is_training=True, train_years=train_years)
    
    # Train model
    print("Training model...")
    predictor = SalesPredictor()
    predictor.train(X_train, y_train, countries)
    
    # Save model
    print("Saving model...")
    predictor.save_model()
    
    print("Training completed!")
    
    # Run analysis
    print("\nGenerating analysis and visualizations...")
    from analyze import analyze_data
    analyze_data()

if __name__ == "__main__":
    main()
