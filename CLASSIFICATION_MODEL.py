#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:36:36 2023

@author: luyifan
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
import numpy as np
pd.options.mode.chained_assignment = None  # clear warning
kickstarter_data = pd.read_excel('/Users/luyifan/Desktop/Kickstarter.xlsx')
test_kickstarter_data = pd.read_excel('/Users/luyifan/Desktop/Kickstarter-Grading-Sample.xlsx')

# Preprocess the data for classification

# Encode target variable + remove invliad target
kickstarter_data = kickstarter_data[kickstarter_data['state'].isin(['successful', 'failed'])]
test_kickstarter_data = test_kickstarter_data[test_kickstarter_data['state'].isin(['successful', 'failed'])]

# Selecting predictors that are available at the time of project's launch
# ASSUMPTION: some of the very useful features like staff_pick are assumed not to be available at launch time base on
# online information
predictors = [
    'goal', 'country', 'currency', 'disable_communication', 'create_to_launch_days',
    'deadline', "created_at", 'static_usd_rate', 'category', 'name_len', 'name_len_clean',
    'blurb_len', 'blurb_len_clean',  "deadline_weekday", "created_at_weekday",
    'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr',
    'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
    'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr',
    'create_to_launch_days',
    'launch_to_deadline_days',
]

print(len(predictors))


X = kickstarter_data[predictors]
y = kickstarter_data['state']

X_grading = test_kickstarter_data[predictors]
y_grading = test_kickstarter_data['state']

# Handling missing values with mean and mode
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

for num_col in numerical_columns:
    median_val = X[num_col].median()
    X[num_col].fillna(median_val, inplace=True)
    
    median_val = X_grading[num_col].median()
    X_grading[num_col].fillna(median_val, inplace=True)

for cat_col in categorical_columns:
    mode_val = X[cat_col].mode()[0]
    X[cat_col].fillna(mode_val, inplace=True)
    mode_val = X_grading[cat_col].mode()[0]
    X_grading[cat_col].fillna(mode_val, inplace=True)


# Preprocess categorical values
X = pd.get_dummies(X, columns=['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday'], drop_first=True)
X_grading = pd.get_dummies(X_grading, columns=['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday'], drop_first=True)

# Preprocess timestamp features
timestamp_columns = ['deadline', 'created_at']
for time_col in timestamp_columns:
    X[f'{time_col}_year'] = X[time_col].dt.year
    X[f'{time_col}_month'] = X[time_col].dt.month
    X[f'{time_col}_day'] = X[time_col].dt.day
X = X.drop(columns=timestamp_columns)

for time_col in timestamp_columns:
    X_grading[f'{time_col}_year'] = X_grading[time_col].dt.year
    X_grading[f'{time_col}_month'] = X_grading[time_col].dt.month
    X_grading[f'{time_col}_day'] = X_grading[time_col].dt.day
X_grading = X_grading.drop(columns=timestamp_columns)

# drop columns that is in X but not in X_grading
columns_to_drop = [col for col in X.columns if col not in X_grading.columns]

X.drop(columns=columns_to_drop, inplace=True)

# Standardize data using MINMAX scaler
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# **Tried PCA but leads to a lower score, so commented out**

# Perform PCA with 95% variance explained
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)
# pca.fit(X_train)
# print("Number of components:", pca.n_components_)
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# cum_explained_var = np.cumsum(pca.explained_variance_ratio_)

# import matplotlib.pyplot as plt
# PC_values = np.arange(1, pca.n_components_ + 1)
# plt.plot(PC_values, cum_explained_var, linestyle='--', linewidth=2, color='blue', label='Cumulative')
# plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label='Individual')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
# plt.legend()
# plt.show()

# pca = PCA(n_components=43)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)
# X_grading = pca.fit_transform(X_grading)

# Feature selection
rf_classifier = RandomForestClassifier(random_state=0)
rfe = RFE(estimator=rf_classifier, step=1)
rfe.fit(X_train, y_train)

# Transform data
X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)
X_grading = rfe.transform(X_grading)



# Hyper parameter tunning
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10, 20, 30],
    'max_features': [10, 15, 20, "auto"]
}


rf_classifier = RandomForestClassifier(random_state=0)

# GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best scores: {grid_search.best_score_}")


# Use best estimator to make predictions on the test set
best_rf_classifier = grid_search.best_estimator_
y_pred = best_rf_classifier.predict(X_test)

# Print measurements
rf_report = classification_report(y_test, y_pred)

print(rf_report)

# Calculate the mean squared error of prediction
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# grading dataset
y_pred_grading = best_rf_classifier.predict(X_grading)
rf_report = classification_report(y_grading, y_pred_grading)

print(rf_report)

# Calculate the mean squared error of prediction
from sklearn.metrics import accuracy_score
print(accuracy_score(y_grading, y_pred_grading))