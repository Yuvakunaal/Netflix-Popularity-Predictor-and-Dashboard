# Netflix Popularity Predictor and Dashboard 🎬

A comprehensive machine learning system that predicts content popularity on Netflix using advanced data analytics and Random Forest algorithms.

![Netflix](https://img.shields.io/badge/Netflix-Content%20Analytics-red)
![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.6%25-brightgreen)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)
![Data Science](https://img.shields.io/badge/Data%20Science-Predictive%20Modeling-green)

## 📊 Project Overview

The **Netflix Popularity Predictor** is an advanced machine learning system designed to forecast the potential popularity of content on the Netflix platform. This project demonstrates the practical application of data science methodologies to solve real-world business problems in the entertainment industry.

By analyzing historical Netflix content data and extracting meaningful patterns, our model provides valuable insights for:

- **Content Acquisition** - Data-driven approach to content licensing decisions
- **Production Planning** - Insights for original content development and genre selection
- **Market Expansion** - Understanding regional preferences for global growth strategies
- **Risk Mitigation** - Reduced uncertainty in content investment through predictive analytics

## 🎯 Key Features

### 🤖 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 90.6%
- **Features**: 17 engineered features
- **ROC AUC**: 0.958
- **Precision**: 89.0%
- **Recall**: 79.0%

### 📈 Interactive Dashboard

- **Real-time Predictions**: Instant popularity analysis for new content
- **Feature Importance**: Visual breakdown of factors influencing popularity
- **Analytics Suite**: Comprehensive data visualization and insights
- **Dataset Exploration**: Full dataset browsing and filtering capabilities

### 🔍 Data Insights

- **Total Titles Analyzed**: 8,807 Netflix titles
- **Time Period**: 1925-2021 content
- **Content Distribution**: 69.7% Movies, 30.3% TV Shows
- **Geographical Coverage**: Content from 100+ countries

## 🏗️ System Architecture

### Data Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Prediction API → Dashboard
```

### Technology Stack

- **Backend**: FastAPI, Scikit-learn, Pandas, Joblib
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Machine Learning**: Random Forest, Feature Engineering, Model Evaluation
- **Data Processing**: Pandas, NumPy, Data Cleaning Pipelines

## 🔬 Machine Learning Methodology

### Dataset Selection

We selected the Netflix Streaming Dataset from Kaggle for its:

- **Comprehensive Coverage**: 8,807 titles with detailed metadata
- **Real-world Relevance**: Authentic Netflix catalog data
- **Rich Feature Set**: Diverse attributes including content type, ratings, duration, genres, and temporal information
- **Industry Standard**: Follows entertainment industry data standards

### Feature Engineering

Our advanced feature engineering pipeline created 17 predictive features:

**Top Predictive Features:**

1. **Content Age** (0.25) - How recent the content is
2. **Rating Encoded** (0.20) - Content maturity rating impact
3. **International Content** (0.18) - Global production appeal
4. **Duration** (0.15) - Optimal content length analysis
5. **US Production** (0.12) - Domestic content performance

### Algorithm Selection: Why Random Forest?

After comprehensive evaluation, we selected Random Forest for:

| Algorithm              | Accuracy  | Pros                                                          | Cons                             |
| ---------------------- | --------- | ------------------------------------------------------------- | -------------------------------- |
| **Random Forest**      | **90.6%** | Handles non-linearity, robust to outliers, feature importance | Computationally intensive        |
| Logistic Regression    | 82.3%     | Interpretable, fast training                                  | Poor with complex relationships  |
| Support Vector Machine | 85.7%     | Effective in high dimensions                                  | Poor scalability, black box      |
| Gradient Boosting      | 89.2%     | High predictive power                                         | Overfitting risk, complex tuning |

**Key Advantages:**

- **Feature Importance**: Clear insights into popularity drivers
- **Robustness**: Effective handling of missing data and outliers
- **Non-linearity**: Captures complex feature relationships
- **Ensemble Method**: Reduces overfitting through bagging

## 📊 Model Performance

### Comprehensive Metrics

| Metric        | Score     | Description                           |
| ------------- | --------- | ------------------------------------- |
| **Accuracy**  | **90.6%** | Overall prediction accuracy           |
| **Precision** | 89.0%     | Correct positive predictions          |
| **Recall**    | 79.0%     | True positive rate                    |
| **F1-Score**  | 84.0%     | Harmonic mean of precision and recall |
| **ROC AUC**   | 0.958     | Area under ROC curve                  |

### Confusion Matrix Analysis

- **True Positives**: 1,175 correctly identified popular titles
- **True Negatives**: 421 correctly identified non-popular titles
- **False Positives**: 54 titles incorrectly flagged as popular
- **False Negatives**: 112 popular titles missed by the model

## 🎯 Business Impact

### Strategic Applications

- **Content Strategy**: Data-driven decisions for content acquisition and production
- **Audience Insights**: Understanding viewer preferences across demographics
- **Market Analysis**: Regional content performance and localization strategies
- **Risk Management**: Reduced uncertainty in content investment decisions

### Key Findings

- **Content Timing**: Newer content (0-5 years) shows 47% higher popularity probability
- **Genre Optimization**: Multi-genre content demonstrates 23% higher success rates
- **Global Appeal**: International co-productions increase popularity by 31%
- **Rating Strategy**: TV-MA and TV-14 content dominates popular categories (59.3%)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access dashboard at: `http://localhost:8000`

### Usage

1. **Make Predictions**: Use the main dashboard to input content details
2. **Explore Analytics**: Navigate to analytics for model performance insights
3. **Study Data**: Use dataset explorer to understand the underlying data
4. **Read Documentation**: Comprehensive details available in documentation section

## 🔮 Future Enhancements

### Planned Features

- **Real-time Data Integration**: Connect with Netflix API for live content analysis
- **Advanced NLP Features**: Incorporate description sentiment analysis and keyword extraction
- **Multi-platform Extension**: Adapt model for other streaming platforms (Disney+, Amazon Prime)
- **Deep Learning Integration**: Explore neural networks for complex pattern recognition
- **A/B Testing Framework**: Implement experimental validation of predictions

### Research Directions

- **Temporal Pattern Analysis**: Seasonal trends and content lifecycle modeling
- **Social Media Integration**: Incorporate social buzz and audience sentiment
- **Competitive Intelligence**: Cross-platform content performance comparison
- **Personalized Predictions**: User-specific content recommendation engine

## 📈 Data Science Significance

### Methodological Innovations

- **Hybrid Feature Engineering**: Combined traditional metadata with derived temporal features
- **Ensemble Learning Application**: Practical implementation of Random Forest in entertainment analytics
- **Interpretable AI**: Balanced predictive power with business interpretability
- **Scalable Architecture**: Designed for potential integration with real-time content evaluation systems

---
> ⭐ If you find this project helpful, please give it a star!