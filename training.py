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

# Assuming X_selected and y_train are already defined
# Split the data into 90% training and 10% testing
X_train, X_test, y_train_, y_test_ = train_test_split(X_selected, y_train, test_size=0.1, random_state=32)

# Define GPR models
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

# Fit the stacking model on the training data
stacked_model.fit(X_train, y_train_.to_numpy().reshape(-1))

# Make predictions on the test set
y_pred = stacked_model.predict(X_test)

# Evaluate the model using R² score
r2 = r2_score(y_test_, y_pred)
print("R² score on the test set:", r2)