import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import joblib
import numpy as np
import pandas as pd

class SalesPredictor:
    def __init__(self):
        self.models = {}  # Dictionary to store one model per country
        self.base_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.store_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
    
    def train(self, X, y, countries):
        """Train separate models for each country"""
        X = X.copy()  # Create a copy to avoid modifying the original
        
        # Encode store and product globally if not already encoded
        if 'store_encoded' not in X.columns and 'store' in X.columns:
            X['store_encoded'] = self.store_encoder.fit_transform(X['store'])
        if 'product_encoded' not in X.columns and 'product' in X.columns:
            X['product_encoded'] = self.product_encoder.fit_transform(X['product'])
        
        # Train a separate model for each country
        for country in countries:
            print(f"\nTraining model for {country}")
            
            # Filter data for this country
            country_mask = X['country'] == country
            X_country = X[country_mask].copy()
            y_country = y[country_mask]
            
            # Drop non-numeric columns and ensure encoding
            cols_to_drop = ['country']
            if 'country_month' in X_country.columns:
                cols_to_drop.append('country_month')
            if 'country_dayofweek' in X_country.columns:
                cols_to_drop.append('country_dayofweek')
            if 'store' in X_country.columns:
                cols_to_drop.append('store')
            if 'product' in X_country.columns:
                cols_to_drop.append('product')
            
            X_country = X_country.drop(cols_to_drop, axis=1)
            
            # Ensure all remaining columns are numeric
            numeric_cols = X_country.select_dtypes(include=[np.number]).columns
            X_country = X_country[numeric_cols]
            
            # Create and train a new model for this country
            self.models[country] = clone(self.base_model)
            self.models[country].fit(X_country, y_country)
            
            # Print feature importance for this country
            feature_importance = pd.DataFrame({
                'feature': X_country.columns,
                'importance': self.models[country].feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop features for {country}:")
            print(feature_importance.head())
        
    def predict(self, X):
        """Make predictions using country-specific models"""
        predictions = []
        X = X.copy()  # Create a copy to avoid modifying the original
        
        # Encode store and product using the fitted encoders if not already encoded
        if 'store_encoded' not in X.columns and 'store' in X.columns:
            X['store_encoded'] = self.store_encoder.transform(X['store'])
        if 'product_encoded' not in X.columns and 'product' in X.columns:
            X['product_encoded'] = self.product_encoder.transform(X['product'])
        
        # Get unique countries in the test set
        countries = X['country'].unique()
        
        for country in countries:
            # Filter data for this country
            country_mask = X['country'] == country
            X_country = X[country_mask].copy()
            
            # Drop non-numeric columns and ensure encoding
            cols_to_drop = ['country']
            if 'country_month' in X_country.columns:
                cols_to_drop.append('country_month')
            if 'country_dayofweek' in X_country.columns:
                cols_to_drop.append('country_dayofweek')
            if 'store' in X_country.columns:
                cols_to_drop.append('store')
            if 'product' in X_country.columns:
                cols_to_drop.append('product')
            
            X_country = X_country.drop(cols_to_drop, axis=1)
            
            # Ensure all remaining columns are numeric
            numeric_cols = X_country.select_dtypes(include=[np.number]).columns
            X_country = X_country[numeric_cols]
            
            # Make predictions using the country-specific model
            if country in self.models:
                country_predictions = self.models[country].predict(X_country)
                predictions.extend(country_predictions)
            else:
                raise ValueError(f"No trained model found for country: {country}")
        
        return np.array(predictions)
    
    def save_model(self, filepath='models/sales_predictor.joblib'):
        """Save all country-specific models"""
        model_data = {
            'models': self.models,
            'store_encoder': self.store_encoder,
            'product_encoder': self.product_encoder
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath='models/sales_predictor.joblib'):
        """Load all country-specific models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.store_encoder = model_data['store_encoder']
        self.product_encoder = model_data['product_encoder']
