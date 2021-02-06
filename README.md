# Telecom-Customer-churn-Prediction
This model predict the telecom voluntary customer churn, built using good techniques like SMOTE, RFE in data preprocessing and good supervised ML algo - Logistic regression, KNN, Decision tree, Random forest.

Steps followed:
1. Importing all the necessary/required modules of Python
2. Loading the Train & Test datasets - "Telecom" and "Telecom_test"
3. Overview of both datasets in terms of statistics for business decisions like mean, median, mode, variation, range, std deviation, skewness, kurtosis, distribution
4. Exploratory Data analysis - Univariate analysis of each features by boxplot, barplot, histogram. Bivariate analysis by scatter plot, pairplot, heatmap
5. Correlation check for further data insight like here we observed that Features - Total day minutes, Total day charge, Customer service calls are looking highly correlated with Dependent variable- Churn
6. SMOTE technique used for dataset imbalance treatment upto 70:30 ratio
7. Further divide the datasets into dependent & independent features to apply the algos of scikit learn module
8. Feature selection - Here we used the RFE technique to select the top contributing features
9. Feature scalling by Standardscaler, fit_transform on train dataset and transform to test dataset to avoid the data leakage
10. Model building by using Logistic regression, KNN, DecisiontreeClassifier, Random forest algorithm
11. Cross validation method used for sampling in model
12. Model Evaluation by confusion matrix, Accuracy, precision, recall , roc_auc score, ROC-AUC curve 
13. Random Forest gave the best output with 96% accuracy and 88% area under the ROC curve
