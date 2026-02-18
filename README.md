## Project Overview

This project predicts stress levels using a machine learning pipeline built on top of a behavioral dataset that combines smartphone usage, productivity, sleep, and lifestyle factors. The models include:

- Generalized Additive Model (GAM)
- Deep Neural Network (PyTorch MLP)
- Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)
- An ensemble regressor combining all models

The pipeline also includes:

- Data drift detection using Evidently
- Fairness analysis using Fairlearn (by gender and age groups)
- Model interpretation via partial dependence plots and tree-based feature importances
- Experiment tracking and artifact logging with MLflow

## Dataset

- **Title:** Screen Time, Sleep & Stress Analysis Dataset  
- **Source:** https://www.kaggle.com/datasets/amar5693/screen-time-sleep-and-stress-analysis-dataset/data  
- **Subtitle:** ML-ready dataset analyzing smartphone usage and productivity.  
- **Rows:** 50,000 user records  

### Features

- **User_ID** – Unique identifier  
- **Age** – User age (18–60)  
- **Gender** – Male, Female, Other  
- **Occupation** – Student, Professional, Freelancer, Business Owner  
- **Device_Type** – Android / iOS  
- **Daily_Phone_Hours** – Average daily phone usage  
- **Social_Media_Hours** – Daily time spent on social media  
- **Work_Productivity_Score** – Productivity score (1–10)  
- **Sleep_Hours** – Average sleep duration  
- **Stress_Level** – Stress rating (1–10)  
- **App_Usage_Count** – Number of apps used daily  
- **Caffeine_Intake_Cups** – Daily caffeine consumption  
- **Weekend_Screen_Time_Hours** – Screen time during weekends  

### Properties

- No missing values  
- Clean, structured tabular format  
- Balanced categorical features  
- Suitable for:
  - Stress level prediction (regression)
  - Stress or productivity classification
  - Behavioral clustering
  - Sleep–stress impact modeling
  - Device-based productivity comparison

## How This Project Uses the Dataset

- Treats **Stress_Level** as the primary regression target.  
- Uses the remaining columns as predictors after preprocessing in `utils.py`.  
- Trains multiple models defined in `models.py` and evaluates them in `main.py`.  
- Tracks experiments and logs artifacts (models, plots, drift report, fairness metrics) to MLflow.  

