#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:59:41 2023

@author: luyifan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


kickstarter_data = pd.read_excel('/Users/luyifan/Desktop/Kickstarter.xlsx')

# Encode target variable + remove invliad target
kickstarter_data = kickstarter_data[kickstarter_data['state'].isin(['successful', 'failed'])]
# remove columns that is not useful for clustering
columns_to_remove = ["name", "id"]
kickstarter_data = kickstarter_data.drop(columns=columns_to_remove)


# Encode categorical columns
categorical_columns = ['state', 'country', 'currency', 'category', 'deadline_weekday',
       'state_changed_at_weekday', 'created_at_weekday',
       'launched_at_weekday', 'disable_communication', 'staff_pick', 
       'spotlight']
kickstarter_data = pd.get_dummies(kickstarter_data, columns=categorical_columns, drop_first=True)
print(kickstarter_data.columns)
print(f"There are {len(kickstarter_data)} number of rows in the dataset")


# Preprocess timestamp features
timestamp_columns = ['deadline', 'created_at', 'state_changed_at', 'launched_at']
for time_col in timestamp_columns:
    kickstarter_data[f'{time_col}_year'] = kickstarter_data[time_col].dt.year
    kickstarter_data[f'{time_col}_month'] = kickstarter_data[time_col].dt.month
    kickstarter_data[f'{time_col}_day'] = kickstarter_data[time_col].dt.day
kickstarter_data = kickstarter_data.drop(columns=timestamp_columns)

# Handling missing values with mean and mode
numerical_columns = kickstarter_data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = kickstarter_data.select_dtypes(include=['object']).columns
for num_col in numerical_columns:
    median_val = kickstarter_data[num_col].median()
    kickstarter_data[num_col] = kickstarter_data[num_col].fillna(median_val)

for cat_col in categorical_columns:
    mode_val = kickstarter_data[cat_col].mode()[0]
    kickstarter_data[cat_col] = kickstarter_data[cat_col].fillna(mode_val)


timestamp_columns = kickstarter_data.select_dtypes(include=['datetime64']).columns



# Scale the features
scaler = StandardScaler()
kickstarter_data_scaled = scaler.fit_transform(kickstarter_data)


# Check for missing values
missing_values = kickstarter_data_scaled[np.isnan(kickstarter_data_scaled)]
if len(missing_values) > 0:
    print("Warning: There are missing values in the scaled dataset.")
    print(missing_values)

# Check for infinite values
infinite_values = kickstarter_data_scaled[np.isinf(kickstarter_data_scaled)]
if len(infinite_values) > 0:
    print("Warning: There are infinite values in the scaled dataset.")
    print(infinite_values)

kickstarter_data_scaled[np.isnan(kickstarter_data_scaled)] = np.nanmedian(kickstarter_data_scaled)
kickstarter_data_scaled[np.isinf(kickstarter_data_scaled)] = np.nanmedian(kickstarter_data_scaled)


# Determine the optimal number of clusters using the elbow method
withinss = []
maxk = 15
for i in range(2, maxk):
    print(f"i = {i}")
    kmeans = KMeans(n_clusters=i, random_state=0)
    model = kmeans.fit(kickstarter_data_scaled)
    withinss.append(model.inertia_)


# Create a plot to visualize the elbow method
plt.plot(range(2, maxk), withinss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()


chosen_k = 12
kmeans = KMeans(n_clusters=chosen_k)
model = kmeans.fit(kickstarter_data_scaled)
labels = model.labels_

# Assign cluster labels to the dataset
kickstarter_data['cluster'] = labels


# For visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(kickstarter_data_scaled)
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=labels, cmap='rainbow')
plt.title('clustered data points classified in 2D space')
plt.show()

# Analyze Clusters
cluster_statistics = kickstarter_data.groupby('cluster').mean()
print(cluster_statistics)
print(cluster_statistics["state_successful"])

cluster_size = kickstarter_data['cluster'].value_counts()
print(cluster_size)

cluster_centroids = kmeans.cluster_centers_

# Write cluster statistics to a text file
with open('cluster_statistics.txt', 'w') as file:
    file.write(",".join(kickstarter_data.columns))
    file.write("\n\nCluster Centroids:\n")
    file.write(np.array2string(cluster_centroids, separator=', '))


import seaborn as sns

sns.boxplot(x='cluster', y="state_successful", data=kickstarter_data)
plt.title('state_successful value in all clusters')
plt.show()
for feature in kickstarter_data.columns:
    if feature not in ["id", "name"]:
        print(feature)
        sns.boxplot(x='cluster', y=feature, data=kickstarter_data)
        plt.title(f'{feature} value in all clusters')
        plt.show()