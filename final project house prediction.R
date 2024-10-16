# Load necessary libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(lubridate)
library(corrplot)
library(randomForest)
library(e1071)
library(gridExtra)

###Data Loading
house = read.csv("D:\\Semester 8\\SEN4103 (1) Data Analysis with R\\kc_house_data.csv")


summary(house)
str(house)
head(house,3)
glimpse(house)#more detailed

###Data Cleaning
colSums(is.na(house)) #check if we have missing value in column


house$date = gsub("T000000"," ",house$date)

# Convert 'date' to Date type and extract 'year_sold' and 'month_sold'
house = house |>
  mutate(date = as.Date(date, format="%Y%m%d"),
         year_sold = year(date),
         month_sold = month(date))

# Check for duplicates and remove if any
house = house |> distinct()

# Drop the original 'date' column
house = house |> select(-date)

# Convert categorical variables to factors
house <- house %>%
  mutate(across(c(waterfront, view, condition, grade, floors), as.factor))

glimpse(house)

###Data Manipulation
# Rename columns
house <- house %>%
  rename(Latitude = lat, Longitude = long)

# Create a new column for the age of the house at the time of sale
house = house |>
  mutate(house_age = year_sold - yr_built,
         renovated_age = ifelse(yr_renovated == 0, house_age, year_sold - yr_renovated))

data_filtered_iqr <- house %>%
  mutate(
    lower_bound = quantile(price, 0.25) - 1.5 * IQR(price),
    upper_bound = quantile(price, 0.75) + 1.5 * IQR(price)
  ) %>%
  filter(price >= lower_bound & price <= upper_bound) %>%
  select(-lower_bound, -upper_bound)  # Remove the temporary columns

# Summarize data by zipcode
zipcode_summary <- house %>%
  group_by(zipcode) %>%
  summarize(mean_price = mean(price), 
            median_price = median(price),
            mean_sqft_living = mean(sqft_living),
            count = n()) %>%
  arrange(desc(mean_price))

# Calculate total sales per zip code
total_sales_by_zipcode <- house %>%
  group_by(zipcode) %>%
  summarize(total_sales = n())

# Get top 10 zipcodes
top_zipcodes <- total_sales_by_zipcode %>%
  slice_max(total_sales, n = 10) %>%
  pull(zipcode) # Extract zipcodes as vector

# Calculate counts for top zipcodes
zipcode_view_counts <- house %>%
  filter(zipcode %in% top_zipcodes) %>%
  group_by(zipcode, view) %>%
  summarize(count = n())

# Calculate total sales per view
total_sales_by_view <- house %>%
  group_by(view) %>%
  summarize(total_sales = n())

###Normalization
# Normalize numerical features
num_features <- c("sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15", "house_age", "renovated_age")

preProc <- preProcess(house[, num_features], method = c("center", "scale"))
house[, num_features] <- predict(preProc, house[, num_features])

###Data exploration
# Create the plot
ggplot(zipcode_view_counts, 
       aes(x = reorder(zipcode, -count), 
           y = count, 
           fill = view)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = count), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  labs(title = "Number of Sales by View (Top 10 Zipcodes)", 
       x = "Zipcode", y = "Number of Sales") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Create the plot
ggplot(total_sales_by_view, aes(x = factor(view), y = total_sales, fill = factor(view))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = total_sales), vjust = -0.5, size = 3) +
  labs(title = "Total Number of Sales by View",
       x = "View", y = "Total Sales") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") # Optional: add a color palette


# Ensure all selected columns for the correlation matrix are numeric
cor_matrix_data <- house %>%
  select(price, bedrooms, bathrooms, floors, condition, grade, house_age) %>%
  mutate(across(everything(), as.numeric))

# Calculate the correlation matrix
cor_matrix <- cor(cor_matrix_data)
corrplot(cor_matrix, method = "circle")

# Plot for the original house
p1 <- ggplot(house, aes(x = price)) +
  geom_histogram(fill = "skyblue", alpha = 0.6) +
  labs(title = "Original Prices") +
  theme_minimal()

# Plot for the filtered house (without outliers)
p2 <- ggplot(data_filtered_iqr, aes(x = price)) +
  geom_histogram(fill = "salmon", alpha = 0.6) +
  labs(title = "Filtered Prices (No Outliers)") +
  theme_minimal()

# Display the plots side-by-side
grid.arrange(p1, p2,  ncol = 2)

###ANOVA test
# ANOVA test to see if there are significant differences
anova_results <- aov(price ~ factor(bedrooms) + factor(bathrooms) + factor(floors) + factor(waterfront) + factor(view) + factor(condition) + factor(grade), data = house)
summary(anova_results)

###Machine Learning Techniques
library(doParallel)

# Assuming data is already normalized
# Splitting the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(house$price, p = .8, list = FALSE, times = 1)
trainData <- house[trainIndex,]
testData  <- house[-trainIndex,]

# Detect the number of cores
cores <- detectCores()

# Register the parallel backend
cl <- makeCluster(cores)
registerDoParallel(cl)

# Set up trainControl with 5-fold cross-validation
control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Train a Random Forest model with 5-fold cross-validation and fewer trees
rf_model <- train(price ~ ., data = trainData, method = "rf", trControl = control, ntree = 50, tuneLength = 3, nodesize = 5)

# Predict on the test set
rf_predictions <- predict(rf_model, testData)

# Model evaluation for Random Forest
rf_results <- postResample(rf_predictions, testData$price)
print(rf_results)

# Train an SVM model with 5-fold cross-validation and parallel processing
svm_model <- train(price ~ ., data = trainData, method = "svmRadial", trControl = control, tuneLength = 3)

# Predict on the test set
svm_predictions <- predict(svm_model, testData)

# Model evaluation for SVM
svm_results <- postResample(svm_predictions, testData$price)
print(svm_results)

# Train a Linear Regression model with 5-fold cross-validation
lm_model <- train(price ~ ., data = trainData, method = "lm", trControl = control)

# Predict on the test set
lm_predictions <- predict(lm_model, testData)

# Model evaluation for Linear Regression
lm_results <- postResample(lm_predictions, testData$price)
print(lm_results)

# Stop the parallel backend
stopCluster(cl)
registerDoSEQ()



###Final Visualization
# Actual vs Predicted Prices for Random Forest
ggplot() +
  geom_point(aes(x = testData$price, y = rf_predictions), alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Actual vs Predicted Prices (Random Forest)", x = "Actual Prices", y = "Predicted Prices")

# Feature importance for Random Forest
rf_importance <- varImp(rf_model, scale = FALSE)
ggplot(rf_importance)

# Actual vs Predicted Prices for SVM
ggplot() +
  geom_point(aes(x = testData$price, y = svm_predictions), alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "blue") +
  labs(title = "Actual vs Predicted Prices (SVM)", x = "Actual Prices", y = "Predicted Prices")

# Actual vs Predicted Prices for Linear Regression
ggplot() +
  geom_point(aes(x = testData$price, y = lm_predictions), alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "green") +
  labs(title = "Actual vs Predicted Prices (Linear Regression)", x = "Actual Prices", y = "Predicted Prices")






