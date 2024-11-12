import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data
from model import SalesPredictor
import joblib
import os

def plot_country_comparisons(train_df):
    """Create comparative bar plots for different variables across countries"""
    
    # Average sales by country
    plt.figure(figsize=(12, 6))
    sns.barplot(data=train_df, x='country', y='num_sold', estimator='mean')
    plt.title('Average Daily Sales by Country')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/visualizations/country_avg_sales.png')
    plt.close()
    
    # Maximum sales by country
    plt.figure(figsize=(12, 6))
    country_max = train_df.groupby('country')['num_sold'].max().reset_index()
    sns.barplot(data=country_max, x='country', y='num_sold')
    plt.title('Maximum Daily Sales by Country')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/visualizations/country_max_sales.png')
    plt.close()
    
    # Sales by store and country
    plt.figure(figsize=(15, 8))
    store_country = train_df.groupby(['country', 'store'])['num_sold'].mean().reset_index()
    sns.barplot(data=store_country, x='country', y='num_sold', hue='store')
    plt.title('Average Sales by Store and Country')
    plt.xticks(rotation=45)
    plt.legend(title='Store', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/country_store_sales.png')
    plt.close()
    
    # Sales by product and country
    plt.figure(figsize=(15, 8))
    product_country = train_df.groupby(['country', 'product'])['num_sold'].mean().reset_index()
    sns.barplot(data=product_country, x='country', y='num_sold', hue='product')
    plt.title('Average Sales by Product and Country')
    plt.xticks(rotation=45)
    plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/country_product_sales.png')
    plt.close()
    
    # Monthly sales patterns by country
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    plt.figure(figsize=(15, 8))
    monthly_country = train_df.groupby(['country', 'month'])['num_sold'].mean().reset_index()
    sns.barplot(data=monthly_country, x='month', y='num_sold', hue='country')
    plt.title('Average Monthly Sales by Country')
    plt.xlabel('Month')
    plt.ylabel('Average Daily Sales')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/country_monthly_sales.png')
    plt.close()

    # Monthly performance by store (line plot)
    plt.figure(figsize=(15, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    monthly_store = train_df.groupby(['store', 'month'])['num_sold'].mean().reset_index()
    
    # Create line plot with markers
    sns.lineplot(data=monthly_store, x='month', y='num_sold', hue='store', marker='o')
    
    plt.title('Desempeño mensual por tienda', fontsize=12)
    plt.xlabel('Mes', fontsize=10)
    plt.ylabel('Número de ventas', fontsize=10)
    plt.legend(title='Tienda', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/monthly_store_performance.png')
    plt.close()

    # Monthly performance by country (line plot)
    plt.figure(figsize=(15, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    monthly_country = train_df.groupby(['country', 'month'])['num_sold'].mean().reset_index()
    
    # Create line plot with markers
    sns.lineplot(data=monthly_country, x='month', y='num_sold', hue='country', marker='o')
    
    plt.title('Desempeño mensual por país', fontsize=12)
    plt.xlabel('Mes', fontsize=10)
    plt.ylabel('Número de ventas', fontsize=10)
    plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/monthly_country_performance.png')
    plt.close()

    # Monthly performance by store (line plot)
    plt.figure(figsize=(15, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    monthly_store = train_df.groupby(['store', 'month'])['num_sold'].mean().reset_index()
    
    # Create line plot
    sns.lineplot(data=monthly_store, x='month', y='num_sold', hue='store', marker='o')
    
    plt.title('Desempeño mensual por tienda', fontsize=12)
    plt.xlabel('Mes', fontsize=10)
    plt.ylabel('Número de ventas', fontsize=10)
    plt.legend(title='Tienda', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/visualizations/monthly_store_performance.png')
    plt.close()
    
    # Weekend vs Weekday sales by country
    train_df['is_weekend'] = pd.to_datetime(train_df['date']).dt.dayofweek.isin([5, 6])
    plt.figure(figsize=(12, 6))
    weekend_country = train_df.groupby(['country', 'is_weekend'])['num_sold'].mean().reset_index()
    sns.barplot(data=weekend_country, x='country', y='num_sold', hue='is_weekend')
    plt.title('Weekend vs Weekday Average Sales by Country')
    plt.xticks(rotation=45)
    plt.legend(title='Is Weekend', labels=['Weekday', 'Weekend'])
    plt.tight_layout()
    plt.savefig('data/visualizations/country_weekend_sales.png')
    plt.close()

def create_directories():
    """Create necessary directories for outputs"""
    try:
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/visualizations', exist_ok=True)
        # Ensure write permissions
        os.chmod('data/processed', 0o777)
        os.chmod('data/visualizations', 0o777)
    except Exception as e:
        print(f"Warning: Could not create/modify directories: {e}")

def analyze_data():
    """Analyze the training data and model performance"""
    create_directories()
    
    # Load data
    train_df, _ = load_data()
    
    # Generate country comparison visualizations
    print("\nGenerating country comparison visualizations...")
    plot_country_comparisons(train_df)
    X_train, y_train, countries = preprocess_data(train_df, is_training=True)
    
    # Load trained model
    predictor = SalesPredictor()
    predictor.load_model()
    
    # Make predictions on training data
    y_pred = predictor.predict(X_train)
    
    # Get feature importance from the first country's model (as an example)
    first_country = list(predictor.models.keys())[0]
    first_model = predictor.models[first_country]
    
    # Get numeric features only
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': first_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv('data/processed/feature_importance.csv', index=False)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('data/visualizations/feature_importance.png')
    plt.close()
    
    # Plot actual vs predicted scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.tight_layout()
    plt.savefig('data/visualizations/actual_vs_predicted.png')
    plt.close()

    # Compare Random Forest vs Linear Regression predictions over time
    from sklearn.linear_model import LinearRegression
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_features]
    lr_model.fit(X_train_numeric, y_train)
    lr_predictions = lr_model.predict(X_train_numeric)
    
    # Create monthly averages for comparison, excluding 2020
    train_df['predictions_rf'] = y_pred
    train_df['predictions_lr'] = lr_predictions
    
    # Filter out 2020 data
    train_df_filtered = train_df[train_df['year'] != 2020]
    
    monthly_avg = train_df_filtered.groupby(['year', 'month']).agg({
        'num_sold': 'mean',
        'predictions_rf': 'mean',
        'predictions_lr': 'mean'
    }).reset_index()
    
    # Sort by date for proper plotting
    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
    monthly_avg = monthly_avg.sort_values('date')
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    plt.plot(monthly_avg['date'], monthly_avg['num_sold'], 'k-o', 
            label='Actual', linewidth=2, markersize=6)
    plt.plot(monthly_avg['date'], monthly_avg['predictions_rf'], 'g--', 
            label='Random Forest', linewidth=2)
    plt.plot(monthly_avg['date'], monthly_avg['predictions_lr'], 'b--', 
            label='Linear Regression', linewidth=2)
    
    plt.title('Comparación de Modelos: Ventas Mensuales', fontsize=12)
    plt.xlabel('Fecha', fontsize=10)
    plt.ylabel('Ventas', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/visualizations/model_comparison.png')
    plt.close()

    # Monthly sales by year
    plt.figure(figsize=(15, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Group by year and month
    yearly_monthly_sales = train_df.groupby(['year', 'month'])['num_sold'].mean().reset_index()
    
    # Create line plot for each year
    for year in sorted(yearly_monthly_sales['year'].unique()):
        year_data = yearly_monthly_sales[yearly_monthly_sales['year'] == year]
        plt.plot(year_data['month'], year_data['num_sold'], 
                marker='o', label=str(year), linewidth=2)
    
    plt.title('Ventas mensuales por año', fontsize=12)
    plt.xlabel('Mes', fontsize=10)
    plt.ylabel('Número de ventas', fontsize=10)
    plt.legend(title='Año')
    plt.xticks(range(1, 13))  # Show all months
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/visualizations/yearly_monthly_sales.png')
    plt.close()
    
    # Print model performance metrics
    mse = ((y_train - y_pred) ** 2).mean()
    rmse = mse ** 0.5
    mae = abs(y_train - y_pred).mean()
    
    metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE'],
        'Value': [mse, rmse, mae]
    })
    metrics.to_csv('data/processed/model_metrics.csv', index=False)
    print("\nModel Performance Metrics:")
    print(metrics)
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    analyze_data()
