# Heart-Disease-Detection-Using-PySpark-
Heart disease is a very common today this project is contain machine learning models to find out the disease by simple putting the symptoms of the disease.
Heart disease remains aleading cause of mortality globally, necessitating advanced analytical approaches for early detection and prevention. This project focuses onpredicting heart disease using a comprehensive dataset from Kaggle,leveraging PySpark for data processing and machine learning, and table for data visualization. The dataset includes various patient attributes such as age, sex, chestpain type, resting blood pressure, cholesterol levels, fasting blood sugar, maximum heart rate, exercise-induced-angina, ST depression induced by exercise, and more. The analysis employs multiple classification models, including Logistic Regression, Support Vector Machine SVM, Random Forest Classifier(RF), and Gradient-Boosted Trees (GBT) Classifier, to predict the likelihood of heart disease. The Random Forest Classifier achieved the highest accuracy, emphasizing its effectiveness in handling complex medical data. Tableau visualizations provided clear and insightful representations of the data distribution and model performance. By enabling healthcare providers to make informed decisions, the findings of this project can contribute to improved patient outcomes and reduced healthcare costs.
This dataset contains clinical records of patients to predict heart failure. The dataset is composed of various medical attributes that are potentially relevant to the prediction of heart disease, aiming to aid in the development of predictive models.
Dataset is publically available on kaggale(https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).
# Attributes
The dataset consists of the following columns:
1.	Age: The age of the patient (years)
2.	Sex: The sex of the patient (Male or Female)
3.	ChestPainType: Type of chest pain experienced by the patient:
o	TA: Typical Angina
o	ATA: Atypical Angina
o	NAP: Non-Anginal Pain
o	ASY: Asymptomatic
4.	RestingBP: Resting blood pressure (mm Hg)
5.	Cholesterol: Serum cholesterol level (mg/dL)
6.	FastingBS: Fasting blood sugar (1 if fasting blood sugar > 120 mg/dL, 0 otherwise)
7.	RestingECG: Resting electrocardiogram results:
o	Normal
o	ST: Having ST-T wave abnormality
o	LVH: Showing probable or definite left ventricular hypertrophy by Estes’ criteria
8.	MaxHR: Maximum heart rate achieved (bpm)
9.	ExerciseAngina: Exercise-induced angina (Yes or No)
10.	Oldpeak: ST depression induced by exercise relative to rest
11.	ST_Slope: The slope of the peak exercise ST segment:
o	Up: Upsloping
o	Flat: Flat
o	Down: Downsloping
12.	HeartDisease: Target variable indicating whether the patient has heart disease (1: Yes, 0: No)
# Data Characteristics
•	Total Instances: 918
•	Data Types: The dataset contains numerical and categorical variables.
o	Numerical: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
o	Categorical: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, HeartDisease
Key Features and Usage
The dataset is designed for classification tasks, aiming to predict the presence of heart disease in patients based on various clinical attributes. It can be used for:
•	Building and training machine learning models for heart disease prediction.
•	Analyzing the relationships between different medical attributes and heart disease.
•	Visualizing data distributions and correlations to gain insights into heart disease risk factors.
# Applications
•	Healthcare Analytics: Developing predictive models to assist healthcare professionals in early diagnosis and treatment planning for heart disease.
•	Research: Conducting studies to identify significant predictors of heart disease and improving understanding of cardiovascular health.
•	Education: Serving as a learning tool for students and practitioners in data science and healthcare analytics to practice predictive modeling and data visualization techniques.
# 	Data Pre-processing

1.	Data Cleaning: Handled missing values by filling them with the mean/median for numerical values and the mode for categorical values. Removed duplicates and irrelevant columns to ensure the dataset's integrity.
2.	Feature Selection: Identified relevant features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, the slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia type.
3.	Data Transformation: Applied label encoding to categorical features and normalized numerical features to standardize the data for model training.
# 	Feature Engineering

1.	Label Encoding: Converted categorical variables like sex, chest pain type, and thalassemia type into numerical values for model compatibility.
2.	One-Hot Encoding: Used for categorical variables without an ordinal relationship to retain all possible categories.
3.	Normalization: Standardized numerical features to a common scale to improve model performance and convergence during training.
5.3	Machine Learning Models
1.	Logistic Regression
•	Purpose: Used for binary classification to predict the presence or absence of heart failure.
•	Implementation: Utilized Scikit-learn's logistic regression model. Hyperparameters, including regularization strength (C) and solver type, were tuned using grid search and cross-validation.
•	Evaluation: Metrics included accuracy, precision, recall, and F1 score to assess the model's effectiveness. ROC-AUC curve analysis was conducted to evaluate the model's discrimination capability.

2.	Support Vector Machine (SVM)
•	Purpose: Effective for high-dimensional spaces and binary classification tasks.
•	Implementation: Employed Scikit-learn's SVM model with a radial basis function (RBF) kernel. Hyperparameters such as regularization parameter (C) and kernel coefficient (gamma) were optimized using grid search.
•	Evaluation: Model performance was assessed using accuracy, precision, recall, and F1 score. Additionally, the ROC-AUC curve was used to evaluate the model's ability to distinguish between classes.
Random Forest
•	Purpose: An ensemble method that builds multiple decision trees and merges them to improve accuracy and control overfitting.
•	Implementation: Utilized Scikit-learn's Random Forest classifier. Hyperparameters such as the number of trees (n_estimators), maximum depth, and minimum samples split were tuned using grid search and cross-validation.
•	Evaluation: Metrics included accuracy, precision, recall, and F1 score. Feature importance scores were analyzed to identify the most influential features in predicting heart failure.
3.	Gradient Boosting
•	Purpose: An ensemble technique that builds multiple weak learners, typically decision trees, and combines them to improve model accuracy.
•	Implementation: Employed Scikit-learn's Gradient Boosting classifier. Hyperparameters such as learning rate, the number of trees (n_estimators), and maximum depth were optimized using grid search and cross-validation.
•	Evaluation: Model performance was assessed using accuracy, precision, recall, and F1 score. The ROC-AUC curve was analyzed to evaluate the model's discrimination capability.


