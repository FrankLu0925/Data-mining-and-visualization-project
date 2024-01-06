# Kickstarter Campaign Success Analysis

## Overview
This project explores Kickstarter campaign data to predict project success and understand campaign dynamics.

## Objectives
- Predict the likelihood of campaign success using a classification model.
- Categorize campaigns using clustering to identify different campaign characteristics.

## Methodology
1. **Data Description and Feature Engineering:** 
   - Analysis of Kickstarter campaign data.
   - Focus on features known at the launch time.
   - Extensive feature engineering for model accuracy.

2. **Model Selection and Rationale:** 
   - **Classification Model:** Random Forest algorithm with GridSearchCV for hyperparameter tuning. Validated through 5-fold cross-validation.
   - **Clustering Model:** KMeans algorithm with optimal clusters determined by the Elbow Method.

## Results and Conclusion
- The classification model provides insights into factors influencing campaign success.
- The clustering model offers a detailed understanding of various campaign types.
