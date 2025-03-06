# Load necessary libraries
library(dplyr)
library(ggplot2)
library(randomForest)
library(caret)
library(glmnet)
library(lubridate)  # For temporal feature extraction

# Load datasets
store_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/store.csv")
sales_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/train.csv")
test_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/test.csv")

# --------------------------
# Data Cleaning 
# --------------------------
merged_data <- sales_data %>%
  left_join(store_data, by = "Store") %>%
  mutate(
    CompetitionDistance = coalesce(CompetitionDistance, median(CompetitionDistance, na.rm = TRUE)),
    across(where(is.character), ~gsub("[^[:print:]]", "", .)),
    CompetitionOpenSinceYear = as.numeric(CompetitionOpenSinceYear),
    CompetitionOpenSinceMonth = as.numeric(CompetitionOpenSinceMonth),
    StoreType = as.factor(StoreType),
    Assortment = as.factor(Assortment),
    Date = as.Date(Date)
  )

# --------------------------
# Feature Engineering 
# --------------------------
merged_data <- merged_data %>%
  mutate(
    # Competition features
    CompetitionOpenDuration = (2015 - CompetitionOpenSinceYear) + (12 - CompetitionOpenSinceMonth)/12,
    
    # Temporal features
    Year = year(Date),
    Month = month(Date),
    WeekOfYear = week(Date),
    
    # Transformations
    LogSales = log1p(Sales),
    
    # Interaction terms
    Promo_StoreType = Promo * as.numeric(StoreType)
  )

# --------------------------
# EDA 
# --------------------------
# Sales Distribution
ggplot(merged_data, aes(x = Sales)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  labs(title = "Sales Distribution", x = "Sales", y = "Frequency")

# Seasonality Trends
ggplot(merged_data, aes(x = Date, y = Sales)) +
  geom_line() +
  facet_wrap(~ StoreType) +
  labs(title = "Seasonality Trends by Store Type", x = "Date", y = "Sales")

# Time Series Decomposition
ts_data <- ts(merged_data$Sales, frequency = 365)
decomp <- decompose(ts_data)
plot(decomp)

# --------------------------
# Modeling Preparation
# --------------------------
# Create modeling dataset
model_data <- merged_data %>%
  filter(Open == 1) %>%
  select(Sales, LogSales, Promo, StoreType, Assortment, 
         CompetitionDistance, CompetitionOpenDuration,
         Year, Month, WeekOfYear, Promo_StoreType)

# Split data
set.seed(123)
train_index <- createDataPartition(model_data$Sales, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# --------------------------
# Model Implementation
# --------------------------
# Linear Regression
lm_model <- lm(LogSales ~ Promo + StoreType + Assortment + 
               CompetitionDistance + CompetitionOpenDuration + 
               factor(Month) + Promo_StoreType, data = train_data)
lm_predictions <- expm1(predict(lm_model, newdata = test_data))
lm_rmse <- sqrt(mean((lm_predictions - test_data$Sales)^2))

# Random Forest
set.seed(123)
rf_model <- randomForest(
  LogSales ~ Promo + StoreType + Assortment + CompetitionDistance + 
  CompetitionOpenDuration + Month + Promo_StoreType,
  data = train_data,
  ntree = 100
)
rf_predictions <- expm1(predict(rf_model, newdata = test_data))
rf_rmse <- sqrt(mean((rf_predictions - test_data$Sales)^2))

# Lasso Regression
x_train <- model.matrix(
  ~ Promo + StoreType + Assortment + CompetitionDistance + 
  CompetitionOpenDuration + Month + Promo_StoreType,
  data = train_data
)
y_train <- train_data$LogSales

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
x_test <- model.matrix(
  ~ Promo + StoreType + Assortment + CompetitionDistance + 
  CompetitionOpenDuration + Month + Promo_StoreType,
  data = test_data
)
lasso_predictions <- expm1(predict(lasso_model, newx = x_test, s = "lambda.min"))
lasso_rmse <- sqrt(mean((lasso_predictions - test_data$Sales)^2))

# --------------------------
# Model Comparison
# --------------------------
cat("Linear Regression RMSE:", lm_rmse, "\n",
    "Random Forest RMSE:", rf_rmse, "\n",
    "Lasso Regression RMSE:", lasso_rmse, "\n")
```
