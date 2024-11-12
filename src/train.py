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
    # Use 2017-2019 and 2021 data for training (excluding 2020 pandemic year)
    train_years = [2017, 2018, 2019, 2021]
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
