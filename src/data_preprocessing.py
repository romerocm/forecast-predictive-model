import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def validate_dataframe(df):
    """Validate that the dataframe has the required columns"""
    required_columns = ['id', 'date', 'country', 'store', 'product']
    if any(col not in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

def load_data():
    """Load training and test data"""
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    # Validate both dataframes
    validate_dataframe(train_df)
    validate_dataframe(test_df)
    
    return train_df, test_df

def preprocess_data(df, is_training=True, train_years=None):
    """Preprocess the data for model training/prediction"""
    print("\nDataFrame Info:")
    print(df.info())
    print("\nInput columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Add holiday flags (major international holidays)
    df['is_christmas'] = ((df['month'] == 12) & (df['day'] == 25)).astype(int)
    df['is_new_year'] = ((df['month'] == 1) & (df['day'] == 1)).astype(int)
    
    # Create interaction features between country and temporal features
    df['country_month'] = df['country'] + '_' + df['month'].astype(str)
    df['country_dayofweek'] = df['country'] + '_' + df['day_of_week'].astype(str)
    
    # Encode categorical variables that exist in the dataset
    le = LabelEncoder()
    categorical_features = []
    
    # List of possible categorical columns
    cat_columns = ['country', 'store', 'product']  # Removed 'item' and 'category' as they're not in your data
    
    for col in cat_columns:
        if col in df.columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            categorical_features.append(f'{col}_encoded')
    
    # Base features that should always exist
    base_features = [
        'year', 'month', 'day', 'day_of_week', 
        'is_weekend', 'quarter', 
        'is_christmas', 'is_new_year'
    ]
    
    # Filter training data to exclude 2022 if training
    if is_training and train_years:
        df = df[df['year'].isin(train_years)]
        print(f"Training on years: {sorted(df['year'].unique())}")
    
    if is_training:
        features = base_features + categorical_features
        target = 'num_sold'
        print(f"Features to be used: {features}")
        print(f"Target column: {target}")
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe. Available columns: {df.columns.tolist()}")
        # Include features, raw categorical columns, and interaction features
        all_columns = features + cat_columns + ['country_month', 'country_dayofweek']
        return df[all_columns], df[target], df['country'].unique()
    else:
        features = base_features + categorical_features
        print(f"Features to be used: {features}")
        # Include features, raw categorical columns, and interaction features
        all_columns = features + cat_columns + ['country_month', 'country_dayofweek']
        return df[all_columns]
