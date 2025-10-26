import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self):
        """Fill missing values appropriately"""
        # Fill missing director with 'Unknown'
        self.df['director'] = self.df['director'].fillna('Unknown')
        
        # Fill missing cast with 'Unknown'
        self.df['cast'] = self.df['cast'].fillna('Unknown')
        
        # Fill missing country with 'Unknown'
        self.df['country'] = self.df['country'].fillna('Unknown')
        
        # Fill missing date_added with mode
        mode_date = self.df['date_added'].mode()[0]
        self.df['date_added'] = self.df['date_added'].fillna(mode_date)
        
        # Fill missing rating with mode
        mode_rating = self.df['rating'].mode()[0]
        self.df['rating'] = self.df['rating'].fillna(mode_rating)
        
        return self
    
    def clean_dates(self):
        """Convert date_added to datetime and extract features"""
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        self.df['year_added'] = self.df['date_added'].dt.year
        self.df['month_added'] = self.df['date_added'].dt.month
        
        # Fill any missing dates with reasonable defaults
        self.df['year_added'] = self.df['year_added'].fillna(self.df['release_year'])
        self.df['month_added'] = self.df['month_added'].fillna(1)
        
        return self
    
    def clean_duration(self):
        """Extract numerical duration and unit"""
        # For movies: extract minutes
        movie_mask = self.df['type'] == 'Movie'
        self.df.loc[movie_mask, 'duration_min'] = (
            self.df.loc[movie_mask, 'duration']
            .str.extract('(\d+)')[0]
            .astype(float)
        )
        
        # For TV shows: extract number of seasons
        tv_mask = self.df['type'] == 'TV Show'
        self.df.loc[tv_mask, 'num_seasons'] = (
            self.df.loc[tv_mask, 'duration']
            .str.extract('(\d+)')[0]
            .astype(float)
        )
        
        # Fill missing values
        self.df['duration_min'] = self.df['duration_min'].fillna(0)
        self.df['num_seasons'] = self.df['num_seasons'].fillna(0)
        
        return self
    
    def get_clean_data(self):
        """Return cleaned dataframe"""
        return self.df

# Usage function
def clean_netflix_data(df):
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .handle_missing_values()
                 .clean_dates()
                 .clean_duration()
                 .get_clean_data())
    return cleaned_df