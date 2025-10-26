import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def create_target_variable(self):
        """
        Create target variable for popularity
        We'll define popularity based on:
        - Recent content (last 5 years)
        - Content with specific ratings
        - Content from major countries
        """
        # Simple heuristic for popularity (you can modify this)
        conditions = (
            (self.df['release_year'] >= 2015) &  # Recent content
            (self.df['rating'].isin(['TV-MA', 'TV-14', 'R', 'PG-13'])) &  # Popular ratings
            (self.df['country'].str.contains('United States|India|United Kingdom|Canada'))  # Major markets
        )
        
        self.df['is_popular'] = conditions.astype(int)
        return self
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        # Binary encoding for type
        self.df['is_movie'] = (self.df['type'] == 'Movie').astype(int)
        
        # Label encode rating
        rating_encoder = LabelEncoder()
        self.df['rating_encoded'] = rating_encoder.fit_transform(self.df['rating'])
        
        return self
    
    def create_genre_features(self):
        """Extract and encode genre information"""
        # Split listed_in into individual genres
        self.df['genres'] = self.df['listed_in'].str.split(', ')
        
        # Get all unique genres
        all_genres = set()
        for genres in self.df['genres'].dropna():
            all_genres.update(genres)
        
        # Create binary columns for top genres
        top_genres = ['Dramas', 'Comedies', 'Action', 'Thrillers', 'Documentaries']
        for genre in top_genres:
            self.df[f'genre_{genre.lower()}'] = self.df['listed_in'].str.contains(genre).astype(int)
        
        # Number of genres
        self.df['num_genres'] = self.df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        return self
    
    def create_country_features(self):
        """Create features from country information"""
        # Main country (first country if multiple)
        self.df['main_country'] = self.df['country'].str.split(',').str[0]
        
        # Is US content?
        self.df['is_us'] = self.df['country'].str.contains('United States').fillna(False).astype(int)
        
        # Is international? (not US)
        self.df['is_international'] = (~self.df['country'].str.contains('United States')).fillna(True).astype(int)
        
        # Number of countries
        self.df['num_countries'] = self.df['country'].str.count(',') + 1
        self.df['num_countries'] = self.df['num_countries'].fillna(1)
        
        return self
    
    def create_temporal_features(self):
        """Create time-based features"""
        current_year = 2024
        self.df['age'] = current_year - self.df['release_year']
        self.df['years_since_added'] = current_year - self.df['year_added']
        
        return self
    
    def get_engineered_features(self):
        """Return dataframe with all engineered features"""
        return self.df

def create_features(df):
    engineer = FeatureEngineer(df)
    featured_df = (engineer
                  .create_target_variable()
                  .encode_categorical_features()
                  .create_genre_features()
                  .create_country_features()
                  .create_temporal_features()
                  .get_engineered_features())
    return featured_df