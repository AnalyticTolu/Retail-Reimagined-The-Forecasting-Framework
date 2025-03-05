# Load necessary libraries
library(dplyr)
library(ggplot2)
library(randomForest)
library(caret)
library(glmnet)

# Load the dataset
# Adjust the file paths as needed
store_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/store.csv")
sales_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/train.csv")
test_data <- read.csv("/kaggle/input/d/toluwaseomole/rossmann-store-sales-dataset/test.csv")

# Data Cleaning
# Merge store details with sales data
merged_data <- sales_data %>%
  left_join(store_data, by = "Store")

# Handle missing values
merged_data <- merged_data %>%
  mutate(
    CompetitionDistance = ifelse(is.na(CompetitionDistance), median(CompetitionDistance, na.rm = TRUE), CompetitionDistance),
    Promo2SinceWeek = ifelse(is.na(Promo2SinceWeek), 0, Promo2SinceWeek),
    Promo2SinceYear = ifelse(is.na(Promo2SinceYear), 0, Promo2SinceYear)
  )

# Convert categorical variables to factors
merged_data <- merged_data %>%
  mutate(
    StoreType = as.factor(StoreType),
    Assortment = as.factor(Assortment),
    PromoInterval = as.factor(PromoInterval)
  )

# Ensure no missing values in numerical columns
merged_data <- merged_data %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
  mutate_if(is.character, ~ ifelse(is.na(.), "Unknown", .))

# Encode categorical variables for glmnet
merged_data <- merged_data %>%
  mutate(
    StoreType = as.numeric(StoreType),
    Assortment = as.numeric(Assortment),
    PromoInterval = as.numeric(PromoInterval)
  )

# Feature Engineering
# Create a feature for competition open duration
merged_data <- merged_data %>%
  mutate(
    CompetitionOpenDuration = (2015 - CompetitionOpenSinceYear) + (12 - CompetitionOpenSinceMonth) / 12
  )

# Exploratory Data Analysis (EDA)
# Distribution of sales
ggplot(merged_data, aes(x = Sales)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  labs(title = "Sales Distribution", x = "Sales", y = "Frequency")

# Boxplot of sales by store type
ggplot(merged_data, aes(x = as.factor(StoreType), y = Sales)) +
  geom_boxplot() +
  labs(title = "Sales by Store Type", x = "Store Type", y = "Sales")

# Relationship between competition distance and sales
ggplot(merged_data, aes(x = CompetitionDistance, y = Sales)) +
  geom_point(alpha = 0.3) +
  labs(title = "Competition Distance vs Sales", x = "Competition Distance", y = "Sales")

# Split Data into Training and Testing Sets
# Filter only relevant and clean data
clean_data <- merged_data %>%
  filter(Open == 1) %>% # Use only days when stores were open
  select(Sales, Promo, StoreType, Assortment, CompetitionDistance, Promo2) # Relevant columns

set.seed(123)
train_index <- createDataPartition(clean_data$Sales, p = 0.8, list = FALSE)
train_data <- clean_data[train_index, ]
test_data <- clean_data[-train_index, ]

# Ensure no missing values before modeling
train_data <- train_data %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
  mutate_if(is.character, ~ ifelse(is.na(.), "Unknown", .))

test_data <- test_data %>%
  mutate_if(is.numeric, ~ ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
  mutate_if(is.character, ~ ifelse(is.na(.), "Unknown", .))

sum(is.na(train_data)) # Should return 0
sum(is.na(test_data))  # Should return 0

# Model 1 - Linear Regression
lm_model <- lm(Sales ~ Promo + StoreType + Assortment + CompetitionDistance + Promo2, data = train_data)
summary(lm_model)

# Predict and evaluate
lm_predictions <- predict(lm_model, newdata = test_data)
lm_rmse <- sqrt(mean((lm_predictions - test_data$Sales)^2))
cat("Linear Regression RMSE:", lm_rmse, "\n")

# Model 2 - Random Forest
set.seed(123)
rf_model <- randomForest(Sales ~ Promo + StoreType + Assortment + CompetitionDistance + Promo2, data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_rmse <- sqrt(mean((rf_predictions - test_data$Sales)^2))
cat("Random Forest RMSE:", rf_rmse, "\n")

# Model 3 - Lasso Regression
x_train <- as.matrix(train_data %>% select(-Sales))
y_train <- train_data$Sales
x_test <- as.matrix(test_data %>% select(-Sales))
y_test <- test_data$Sales

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, na.action = na.omit)
lasso_predictions <- predict(lasso_model, newx = x_test, s = "lambda.min")
lasso_rmse <- sqrt(mean((lasso_predictions - y_test)^2))
cat("Lasso Regression RMSE:", lasso_rmse, "\n")

# Compare models
cat("Linear Regression vs Random Forest vs Lasso RMSE:", lm_rmse, "vs", rf_rmse, "vs", lasso_rmse, "\n")
