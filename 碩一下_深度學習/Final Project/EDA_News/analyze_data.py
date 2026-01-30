import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Collection
# Reading TXT files
industry_returns = pd.read_csv('DataExample/Industry/12Industry_return_daily.txt', 
                               sep='\t', 
                               parse_dates=True, 
                               index_col=0)  # Assuming first column is date

fama_mapping = pd.read_csv('DataExample/Industry/Fama12Mapping.txt', sep='\t')

# Reading Parquet file
ravenpack_data = pd.read_parquet('DataExample/RavenPack/2000.pqt')

# 2. Data Exploration
def explore_dataset(df, name):
    print(f"\n--- {name} Dataset Exploration ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"First 5 rows:\n{df.head()}")
    
    # For numerical columns, get statistics
    if df.select_dtypes(include=['number']).shape[1] > 0:
        print(f"Numerical statistics:\n{df.describe()}")

# Explore each dataset
explore_dataset(industry_returns, "Industry Returns")
explore_dataset(fama_mapping, "Fama Mapping")
explore_dataset(ravenpack_data, "RavenPack")

# 3. Data Preprocessing
# 3.1. Handle missing values
industry_returns = industry_returns.fillna(method='ffill')  # Forward fill for time series
ravenpack_data = ravenpack_data.dropna(subset=['important_columns'])  # Replace with actual column names

# 3.2. Feature engineering (example)
# Add day of week for time series data if applicable
if isinstance(industry_returns.index, pd.DatetimeIndex):
    industry_returns['day_of_week'] = industry_returns.index.dayofweek

# 3.3. Merge datasets if needed
# Example: Merge industry returns with Fama mapping
# (This depends on the actual structure of your data)
if 'industry_code' in fama_mapping.columns:
    merged_data = pd.merge(
        industry_returns,
        fama_mapping,
        left_on='industry_column',  # Replace with actual column name
        right_on='industry_code',   # Replace with actual column name
        how='left'
    )

# 3.4. Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = ravenpack_data.select_dtypes(include=['number']).columns
ravenpack_data[numeric_cols] = scaler.fit_transform(ravenpack_data[numeric_cols])

# 4. Save processed data
industry_returns.to_csv('processed_industry_returns.csv')
ravenpack_data.to_parquet('processed_ravenpack.pqt')