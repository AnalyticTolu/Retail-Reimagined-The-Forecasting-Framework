# Retail Reimagined: The Forecasting Framework

## A HarvardX Data Science Capstone Project

### Author: Toluwase Omole

## Project Overview

This project, *Retail Reimagined: The Forecasting Framework*, is a comprehensive data science capstone project aimed at improving retail sales forecasting using the **Rossmann Store Sales** dataset. The project applies machine learning techniques to predict store sales, providing actionable insights for business optimization.

## Key Features
- **Data Cleaning & Preprocessing:** Merging datasets, handling missing values, and encoding categorical variables.
- **Exploratory Data Analysis (EDA):** Visualizing sales trends, seasonality, and time series decomposition.
- **Feature Engineering:** Creating new variables such as `CompetitionOpenDuration`, `SalesLag` features, and interaction terms.
- **Predictive Modeling:** Implementing and evaluating models like **Linear Regression, Random Forest, Lasso Regression, and LightGBM**.
- **Model Performance Comparison:** Using **RMSE, MAE, and R-squared** to determine the best model.
- **Business Recommendations:** Providing data-driven insights for optimizing promotions, inventory, and store operations.

## Technologies Used
- **Programming Language:** R
- **Libraries:** dplyr, ggplot2, caret, lubridate, randomForest, LightGBM
- **Data Source:** [Rossmann Store Sales Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data)

## Data Pipeline
1. **Load & Merge Data:** Combining `train.csv`, `test.csv`, and `store.csv`.
2. **Data Cleaning & Transformation:** Handling missing values, encoding categorical features.
3. **EDA & Feature Engineering:** Identifying trends, seasonality, and creating new features.
4. **Modeling & Evaluation:** Training multiple machine learning models and comparing performance.
5. **Insights & Recommendations:** Interpreting results for real-world business impact.

## Results Summary
- **Random Forest** achieved the best performance with the lowest **RMSE**, making it a suitable choice for sales forecasting.
- **Key factors affecting sales:** Promotions, seasonality, store type, and competition duration.
- **Actionable Recommendations:** Optimizing marketing campaigns and inventory planning based on sales trends.

## How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/AnalyticTolu/retail-forecasting.git
   cd retail-forecasting
   ```
2. Install dependencies in R:
   ```r
   install.packages(c("dplyr", "ggplot2", "caret", "lubridate", "randomForest", "lightgbm"))
   ```
3. Run the R script to preprocess data and train models.
   ```r
   source("retail_forecasting.R")
   ```

## Future Work
- Incorporating deep learning models such as LSTMs for time series forecasting.
- Expanding feature engineering with additional external data sources (e.g., economic indicators).
- Deploying the trained model as a web application for real-time forecasting.

## License
This project is open-source and available under the MIT License.

---
*For more details, refer to the full project report.*

