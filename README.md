# Sales Forecasting Project

This project implements a machine learning system for forecasting sales across different countries, stores, and products.

## Project Structure

```
.
├── data/
│   ├── raw/           # Place input CSV files here
│   ├── processed/     # Generated intermediate files
│   └── visualizations/# Generated plots and charts
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for analysis
└── src/              # Source code
    ├── analyze.py    # Analysis and visualization
    ├── model.py      # ML model implementation
    ├── predict.py    # Generate predictions
    └── train.py      # Model training
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install Docker and docker-compose if not already installed
3. Place your input data files in `data/raw/`:
   - `train.csv` - Training data
   - `test.csv` - Test data for predictions

## Usage

The project uses Docker for reproducible execution:

```bash
# Make the run script executable
chmod +x run.sh

# Train the model
./run.sh train

# Generate analysis & visualizations 
./run.sh analyze

# Generate predictions
./run.sh predict
```

All outputs will be saved to:
- `data/processed/` - Intermediate files and predictions
- `data/visualizations/` - Generated plots and charts

## Data Requirements

Input CSV files should contain these columns:
- id: Unique identifier
- date: Date in YYYY-MM-DD format
- country: Country code
- store: Store identifier
- product: Product identifier
- num_sold: Number of units sold (train.csv only)

## Model Details

- Uses Random Forest Regression
- Trains separate models per country
- Features include:
  - Temporal: year, month, day, day of week, etc.
  - Categorical: encoded country, store, product
  - Holiday indicators
  - Country-specific temporal interactions

## Analysis Outputs

The analysis generates various visualizations in `data/visualizations/`:
- Sales comparisons across countries
- Store and product performance
- Temporal patterns
- Model performance metrics
