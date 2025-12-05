# Food Delivery Time Prediction

A machine learning project that predicts whether food delivery will be **fast** or **slow** based on various factors like distance, weather, traffic, and more.

## Dataset

The project uses `Food_Delivery_Time_Prediction.csv` containing 200 orders with 15 features including:
- Location data (Customer & Restaurant coordinates)
- Weather & Traffic conditions
- Order details (cost, priority, time)
- Delivery person experience
- Customer & Restaurant ratings

## Models Implemented

| Model | Test Accuracy |
|-------|---------------|
| Decision Tree | 60% |
| K-Nearest Neighbors | 55% |
| Gaussian Naive Bayes | 53% |

Additionally, the **Apriori algorithm** is applied to discover association rules among categorical features.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python untitled4.py
```

## Key Features

- **Data preprocessing**: Outlier handling using IQR method
- **Feature engineering**: Haversine distance calculation from coordinates
- **Model evaluation**: Cross-validation, ROC curves, Precision-Recall curves
- **Hyperparameter tuning**: GridSearchCV for optimal model parameters