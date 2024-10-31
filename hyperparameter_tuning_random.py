#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from pyod.models.cblof import CBLOF
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import datetime
from sklearn.model_selection import RandomizedSearchCV


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

X_train=X_train.drop("id", axis=1)
y_train=y_train.drop("id", axis=1)

X_train = X_train.fillna(X_train.median())

robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

cblof = CBLOF(n_clusters=4, contamination=0.045, clustering_estimator=KMeans(n_clusters=4, random_state=43), random_state=42)
cblof.fit(X_pca)
cluster_labels = cblof.cluster_labels_  # Cluster labels assigned by CBLOF
outlier_labels = cblof.labels_ 

standard_scaler_post = StandardScaler()
X_scaled = standard_scaler_post.fit_transform(X_train[outlier_labels == 0])
y_train = y_train[outlier_labels == 0]

# Fit SelectKBest to get feature scores
selector = SelectKBest(score_func=f_classif, k=225)  # Use 'all' to get scores for all features
X_selected = selector.fit_transform(X_scaled, y_train)

gpr_matern = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=1.5), random_state=42)
gpr_rbf_white = GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0), random_state=42)

# Create base models, including LightGBM and KNN
base_models = [
    ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', random_state=42)),
    ('GPR Matern', gpr_matern),
    ('GPR RBF + White', gpr_rbf_white),
    ('LightGBM', lgb.LGBMRegressor(random_state=42, verbose=-1)),  # LightGBM model
    ('KNN', KNeighborsRegressor(n_neighbors=1)),  # KNN model
]

# Create stacking regressor
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression()
)


# Define parameter distributions instead of a fixed grid
param_distributions = {
    'XGBoost__n_estimators': [50, 100, 150],
    'XGBoost__learning_rate': [0.2, 0.3, 0.4],
    'XGBoost__max_depth': [4, 6, 8],
    'LightGBM__n_estimators': [50, 100, 150],
    'LightGBM__learning_rate': [0.05, 0.1, 0.2]
    # Uncomment if you'd like to tune GPR hyperparameters
    # 'GPR Matern__alpha': [1e-10, 1e-2],
    # 'GPR RBF + White__alpha': [1e-10, 1e-2]
}

# Wrap stacking regressor in RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=stacked_model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of random parameter combinations to try
    cv=5,  # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,
    verbose=4,
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_selected, y_train.to_numpy().reshape(-1))

# Save results
results_df = pd.DataFrame(random_search.cv_results_)
sorted_results = results_df.sort_values(by="mean_test_score", ascending=False)
sorted_results.to_csv("cv_random_result_" + str(datetime.datetime.now().time()) + ".csv")

# Get the best parameters and R² score
print("Best parameters found:", random_search.best_params_)
print("Best cross-validated R² score:", random_search.best_score_)
print(sorted_results.head(10))
