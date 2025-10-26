import pandas as pd

def load_netflix_data(file_path):
    """Load Netflix dataset from CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Test data loading
    df = load_netflix_data('../data/raw/netflix_titles.csv')
    if df is not None:
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())