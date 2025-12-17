# -*- coding: utf-8 -*-
"""
Food Delivery Time Prediction
Predicts whether food delivery will be fast or slow using ML models.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for saving plots
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             PrecisionRecallDisplay, RocCurveDisplay, roc_curve, auc)
from mlxtend.frequent_patterns import apriori, association_rules

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def load_and_explore_data(filepath):
    """Load dataset and display basic info."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATA LOADING AND EXPLORATION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df

def statistical_summary(df):
    """Generate statistical summary and save visualizations."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe().T)
    
    # Histogram of numerical features
    df.hist(figsize=(12, 10), bins=15, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/01_numerical_distributions.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/01_numerical_distributions.png")
    
    # Correlation with Delivery_Time
    correlation = df.select_dtypes(include=['float64', 'int64']).corr()
    print(f"\nCorrelation with Delivery_Time:\n{correlation['Delivery_Time'].sort_values(ascending=False)}")
    
    correlation['Delivery_Time'].sort_values(ascending=False).plot(kind='bar', color='skyblue', figsize=(10, 6))
    plt.title("Correlation of Features with Delivery_Time")
    plt.ylabel("Correlation Coefficient")
    plt.tight_layout()
    plt.savefig('plots/02_correlation_with_delivery_time.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/02_correlation_with_delivery_time.png")

def handle_outliers(df):
    """Handle outliers using IQR method."""
    print("\n" + "=" * 60)
    print("OUTLIER HANDLING")
    print("=" * 60)
    outlier_cols = ['Order_Cost', 'Tip_Amount', 'Customer_Rating', 'Distance']
    
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
    
    print(f"✓ Outliers capped for: {outlier_cols}")
    return df

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula."""
    R = 6371  # Earth's radius in km
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def extract_lat_lon(location_str):
    """Extract latitude and longitude from string like '(lat, lon)'."""
    try:
        lat_lon = location_str.replace('(', '').replace(')', '').split(',')
        return float(lat_lon[0]), float(lat_lon[1])
    except:
        return None, None

def feature_engineering(df):
    """Calculate distance from coordinates."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df['Customer_latitude'], df['Customer_longitude'] = zip(*df['Customer_Location'].apply(extract_lat_lon))
    df['Restaurant_latitude'], df['Restaurant_longitude'] = zip(*df['Restaurant_Location'].apply(extract_lat_lon))
    
    df['Distance'] = df.apply(lambda row: haversine(
        row['Restaurant_latitude'], row['Restaurant_longitude'],
        row['Customer_latitude'], row['Customer_longitude']
    ), axis=1)
    
    print("✓ Calculated haversine distance from coordinates")
    return df

def prepare_data(df):
    """Prepare data for modeling."""
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    # Create binary target
    avg_time = df['Delivery_Time'].mean()
    df['Delivery_Status'] = np.where(df['Delivery_Time'] <= avg_time, 1, 0)
    print(f"✓ Created binary target (Fast=1 if Delivery_Time <= {avg_time:.2f} min)")
    
    # Features and target
    drop_cols = ['Delivery_Time', 'Order_ID', 'Customer_Location', 'Restaurant_Location',
                 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_Status']
    X = df.drop(drop_cols, axis=1)
    y = df['Delivery_Status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    # Scale numerical columns
    scaler = StandardScaler()
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_processed[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"✓ Training set: {X_train_processed.shape}")
    print(f"✓ Test set: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, df

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Train and evaluate Gaussian Naive Bayes."""
    print("\n" + "=" * 60)
    print("GAUSSIAN NAIVE BAYES")
    print("=" * 60)
    
    model = GaussianNB()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Slow', 'Fast'], yticklabels=['Slow', 'Fast'])
    plt.title('Confusion Matrix - Naive Bayes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/03_confusion_matrix_naive_bayes.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/03_confusion_matrix_naive_bayes.png")
    
    return model

def train_knn(X_train, X_test, y_train, y_test):
    """Train and evaluate K-Nearest Neighbors."""
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS")
    print("=" * 60)
    
    # Find best k
    param_grid = {'n_neighbors': np.arange(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Best k: {best_k}")
    
    model = KNeighborsClassifier(n_neighbors=best_k)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Slow', 'Fast'], yticklabels=['Slow', 'Fast'])
    plt.title(f'Confusion Matrix - KNN (k={best_k})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/04_confusion_matrix_knn.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/04_confusion_matrix_knn.png")
    
    return model

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate Decision Tree."""
    print("\n" + "=" * 60)
    print("DECISION TREE")
    print("=" * 60)
    
    # Find best parameters
    param_grid = {
        'max_depth': np.arange(1, 21),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=50), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    
    model = DecisionTreeClassifier(random_state=50, **grid_search.best_params_)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Slow', 'Fast'], yticklabels=['Slow', 'Fast'])
    plt.title('Confusion Matrix - Decision Tree')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/05_confusion_matrix_decision_tree.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/05_confusion_matrix_decision_tree.png")
    
    return model

def plot_roc_comparison(models, X_test, y_test):
    """Plot ROC curves for all models."""
    print("\n" + "=" * 60)
    print("ROC CURVE COMPARISON")
    print("=" * 60)
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('ROC Curves - All Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/06_roc_comparison.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/06_roc_comparison.png")

def plot_distance_distribution(df):
    """Plot distance distribution by delivery status."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Delivery_Status', y='Distance', data=df)
    plt.title('Distance Distribution by Delivery Status')
    plt.xlabel('Delivery Status (0: Slow, 1: Fast)')
    plt.ylabel('Distance (km)')
    plt.xticks([0, 1], ['Slow', 'Fast'])
    plt.tight_layout()
    plt.savefig('plots/07_distance_by_status.png', dpi=150)
    plt.close()
    print("✓ Saved: plots/07_distance_by_status.png")

def run_apriori(df):
    """Run Apriori algorithm for association rules."""
    print("\n" + "=" * 60)
    print("APRIORI ALGORITHM")
    print("=" * 60)
    
    df_apriori = df.copy()
    categorical_cols = ['Weather_Conditions', 'Traffic_Conditions', 'Order_Priority',
                        'Order_Time', 'Vehicle_Type', 'Delivery_Status']
    df_apriori['Delivery_Status'] = df_apriori['Delivery_Status'].astype(str)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_data = ohe.fit_transform(df_apriori[categorical_cols])
    transactions_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical_cols))
    
    frequent_itemsets = apriori(transactions_df, min_support=0.1, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        print(f"Found {len(rules)} association rules with confidence >= 0.7")
        if len(rules) > 0:
            print(f"\nTop rules:\n{rules.head()}")

def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("FOOD DELIVERY TIME PREDICTION")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_explore_data('Food_Delivery_Time_Prediction.csv')
    statistical_summary(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, df = prepare_data(df)
    
    # Train models
    nb_model = train_naive_bayes(X_train, X_test, y_train, y_test)
    knn_model = train_knn(X_train, X_test, y_train, y_test)
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
    
    # Compare models
    models = {
        'Naive Bayes': nb_model,
        'KNN': knn_model,
        'Decision Tree': dt_model
    }
    plot_roc_comparison(models, X_test, y_test)
    plot_distance_distribution(df)
    
    # Association rules
    run_apriori(df)
    
    print("\n" + "=" * 60)
    print("DONE! All plots saved to 'plots/' directory")
    print("=" * 60)

if __name__ == "__main__":
    main()
