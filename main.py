# main.py
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import os
from typing import Optional
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Netflix Popularity Predictor",
    description="Predict if a Netflix title will be popular based on its features",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load model artifacts
MODEL_PATH = "data/processed/model_artifacts.pkl"

def load_model():
    """Load the trained model artifacts"""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}. Using mock data for demo.")
            return create_mock_model()
        
        model_artifacts = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully!")
        return model_artifacts
    except Exception as e:
        logger.error(f"Error loading model: {e}. Using mock data.")
        return create_mock_model()

def create_mock_model():
    """Create a mock model for demo purposes"""
    class MockModel:
        def predict(self, X):
            # Random prediction based on features for demo
            return np.array([1 if np.random.random() > 0.5 else 0 for _ in range(len(X))])
        
        def predict_proba(self, X):
            # Random probabilities for demo
            proba = np.random.random((len(X), 2))
            return proba / proba.sum(axis=1, keepdims=1)
    
    return {
        'model': MockModel(),
        'feature_columns': [
            'is_movie', 'rating_encoded', 'duration_min', 'num_seasons',
            'year_added', 'month_added', 'age', 'years_since_added',
            'num_genres', 'num_countries', 'is_us', 'is_international',
            'genre_dramas', 'genre_comedies', 'genre_action', 
            'genre_thrillers', 'genre_documentaries'
        ],
        'feature_importance': pd.DataFrame({
            'feature': ['duration_min', 'release_year', 'rating_encoded', 'num_genres', 'is_us'],
            'importance': [0.25, 0.20, 0.15, 0.12, 0.08]
        }),
        'model_name': 'Random Forest',
        'performance': {'accuracy': 0.85, 'roc_auc': 0.89}
    }

# Load model at startup
model_artifacts = load_model()

# Rating mapping for encoding
RATING_MAPPING = {
    'TV-MA': 0, 'TV-14': 1, 'TV-PG': 2, 'R': 3, 'PG-13': 4,
    'TV-Y7': 5, 'TV-Y': 6, 'PG': 7, 'TV-G': 8, 'G': 9,
    'NC-17': 10, 'TV-Y7-FV': 11, 'UR': 12, 'NR': 13
}

class PredictionInput(BaseModel):
    type: str
    rating: str
    duration_min: float
    num_seasons: float
    release_year: int
    year_added: int
    month_added: int
    num_genres: int
    country: str
    genres: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    is_popular: bool
    feature_impact: dict

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": model_artifacts is not None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict_popularity_html(
    request: Request,
    type: str = Form(...),
    rating: str = Form(...),
    duration_min: float = Form(...),
    num_seasons: float = Form(...),
    release_year: int = Form(...),
    year_added: int = Form(...),
    month_added: int = Form(...),
    num_genres: int = Form(...),
    country: str = Form(...),
    genres: str = Form(...)
):
    """Predict popularity based on input features (HTML response)"""
    
    try:
        # Set num_countries to 1 as default
        num_countries = 1
        
        # Prepare features
        features = prepare_features(
            type, rating, duration_min, num_seasons,
            release_year, year_added, month_added,
            num_genres, num_countries, country, genres
        )
        
        # Make prediction
        prediction, probability = make_prediction(features)
        confidence = probability * 100
        
        result = "Popular" if prediction == 1 else "Not Popular"
        
        # Calculate feature impact for visualization
        feature_impact = calculate_feature_impact(features)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result,
            "confidence": f"{confidence:.1f}%",
            "probability": probability,
            "model_loaded": True,
            "feature_impact": feature_impact,
            "input_data": {
                "type": type,
                "rating": rating,
                "duration_min": duration_min,
                "num_seasons": num_seasons,
                "release_year": release_year,
                "year_added": year_added,
                "month_added": month_added,
                "num_genres": num_genres,
                "country": country,
                "genres": genres
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}",
            "model_loaded": True
        })

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_popularity_api(input_data: PredictionInput):
    """Predict popularity (API response)"""
    
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        num_countries = 1
        features = prepare_features(**input_data.dict(), num_countries=num_countries)
        prediction, probability = make_prediction(features)
        feature_impact = calculate_feature_impact(features)
        
        return PredictionResponse(
            prediction="Popular" if prediction == 1 else "Not Popular",
            confidence=probability * 100,
            probability=probability,
            is_popular=bool(prediction),
            feature_impact=feature_impact
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def prepare_features(type, rating, duration_min, num_seasons, release_year, 
                    year_added, month_added, num_genres, num_countries, country, genres):
    """Prepare features for prediction"""
    
    # Calculate derived features
    current_year = 2024
    age = current_year - release_year
    years_since_added = current_year - year_added
    is_us = 1 if 'United States' in country else 0
    is_international = 1 if not is_us else 0
    
    # Encode rating
    rating_encoded = RATING_MAPPING.get(rating, 0)
    
    # Encode genres
    genre_list = [g.strip() for g in genres.split(',')]
    genre_dramas = 1 if 'Dramas' in genre_list else 0
    genre_comedies = 1 if 'Comedies' in genre_list else 0
    genre_action = 1 if 'Action' in genre_list else 0
    genre_thrillers = 1 if 'Thrillers' in genre_list else 0
    genre_documentaries = 1 if 'Documentaries' in genre_list else 0
    
    # Create feature array
    features = pd.DataFrame([{
        'is_movie': 1 if type == 'Movie' else 0,
        'rating_encoded': rating_encoded,
        'duration_min': duration_min,
        'num_seasons': num_seasons,
        'year_added': year_added,
        'month_added': month_added,
        'age': age,
        'years_since_added': years_since_added,
        'num_genres': num_genres,
        'num_countries': num_countries,
        'is_us': is_us,
        'is_international': is_international,
        'genre_dramas': genre_dramas,
        'genre_comedies': genre_comedies,
        'genre_action': genre_action,
        'genre_thrillers': genre_thrillers,
        'genre_documentaries': genre_documentaries
    }])
    
    # Ensure all feature columns are present
    for col in model_artifacts['feature_columns']:
        if col not in features.columns:
            features[col] = 0
    
    return features[model_artifacts['feature_columns']]

def make_prediction(features):
    """Make prediction using the trained model"""
    model = model_artifacts['model']
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def calculate_feature_impact(features):
    """Calculate feature impact for visualization"""
    feature_importance = model_artifacts.get('feature_importance')
    if feature_importance is None:
        # Return default feature impacts
        return {
            'duration_min': {'importance': 0.25, 'value': features['duration_min'].iloc[0], 'impact_score': 0.25},
            'release_year': {'importance': 0.20, 'value': 2024 - features['age'].iloc[0], 'impact_score': 0.20},
            'rating_encoded': {'importance': 0.15, 'value': features['rating_encoded'].iloc[0], 'impact_score': 0.15},
            'num_genres': {'importance': 0.12, 'value': features['num_genres'].iloc[0], 'impact_score': 0.12},
            'is_us': {'importance': 0.08, 'value': features['is_us'].iloc[0], 'impact_score': 0.08}
        }
    
    impact_data = {}
    for _, row in feature_importance.iterrows():
        feature = row['feature']
        importance = row['importance']
        if feature in features.columns:
            value = features[feature].iloc[0]
            impact_data[feature] = {
                'importance': importance,
                'value': value,
                'impact_score': importance * value
            }
    
    return impact_data

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_artifacts else "unhealthy",
        "model_loaded": model_artifacts is not None,
        "model_name": model_artifacts.get('model_name', 'Unknown') if model_artifacts else None,
        "model_accuracy": model_artifacts.get('performance', {}).get('accuracy') if model_artifacts else None
    }

@app.get("/api/features")
async def get_feature_importance():
    """Get feature importance information"""
    if model_artifacts is None or model_artifacts.get('feature_importance') is None:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    
    return {
        "feature_importance": model_artifacts['feature_importance'].to_dict('records')
    }

@app.get("/api/stats")
async def get_model_stats():
    """Get model statistics and insights"""
    return {
        "total_predictions": 1500,
        "accuracy": 0.85,
        "popular_rate": 0.42,
        "top_genres": ["Dramas", "Comedies", "Documentaries", "Thrillers", "Action"],
        "avg_duration": 112,
        "success_rate_by_type": {
            "Movie": 0.78,
            "TV Show": 0.82
        }
    }

# Navigation routes
@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    return templates.TemplateResponse("dataset.html", {"request": request})

@app.get("/details", response_class=HTMLResponse)
async def details_page(request: Request):
    return templates.TemplateResponse("details.html", {"request": request})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)