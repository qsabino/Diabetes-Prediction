#################### 1. Install if needed and load necessary libraries


if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(Metrics)) install.packages("Metrics")
if(!require(knitr)) install.packages("knitr")
if(!require(e1071)) install.packages("e1071")
if(!require(gridExtra))install.packages("gridExtra")
if(!require(pheatmap))install.packages("pheatmap")

library(tidyverse)
library(caret)
library(Metrics)
library(knitr)
library(e1071)
library(gridExtra)
library(pheatmap)


############################ 2. Load and inspect the data


# Controls the number of digits to print when printing numeric values
options(digits = 6)

# Set options to avoid scientific notation
options(scipen = 10)

# Load data
dat <- read_csv("diabetes.csv")

# Data dimension
dim(dat)

# Inspect the structure of the dataset
str(dat)

# View data
View(dat)

# Summary statistics
summary(dat)

# Frequency table of the Outcome
table(dat$Outcome)


############################ 3. Data cleaning


# 3.1. Missing values

# Identify and count any missing values in each column of a data frame dat
missing_values <- colSums(is.na(dat))
print(missing_values) # zero missing values


# 3.2. Zero valuses

# Identify and count any zero values that may require attention.
n_zeros <- sapply(dat, function(x) sum(x==0))
print(n_zeros)
# These non sense zeros in BloodPressure, BMI, Glucose, Insulin, SkinThickness are not actual observations. 
# They may be due to :
# Measurement error: device and testing
# Severe medical condition of the patients
# Data entry mistake
# Calculation error.

# Replace zeros in columns where they don't make sense (e.g., BloodPressure, BMI, Glucose, Insulin, SkinThickness)
# Function to replace zeros with column mean
replace_zeros_with_mean <- function(x) {
  mean_value <- mean(x[x != 0], na.rm = TRUE) # Calculate mean of non-zero values
  x[x == 0] <- mean_value # Replace zeros with the mean value
  return(x)
}

# Columns with nonsense zeros
columns_with_nonsense_zeros <- c("BloodPressure", "BMI", "Glucose", "Insulin", "SkinThickness")

# Apply the function to columns with nonsense zeros using lapply
dat[columns_with_nonsense_zeros] <- lapply(dat[columns_with_nonsense_zeros], replace_zeros_with_mean)


# 3.3. Outlinears

# Identify outlinears
dat %>% select("Insulin") %>% filter(Insulin > 500) %>% nrow() # 9rows

# Elevated insulin levels might occur in individuals with type 2 diabetes or a precursor.
# But this significant high insulin may be due to rare insulin-secreting tumor or out linear

# Remove rows where insulin > 500
dat <- dat[dat$Insulin <= 500, ]


######################### 4. Exploratory Data Analysis (EDA)


# 4.1. Data statistic and visualization of relationships

# Data statistics
summary_df <- dat %>%
  # diff(range(.)) computes the difference between the maximum and minimum values.
  summarise(across(everything(), list(mean = mean, sd = sd, median = median, range = ~ diff(range(.))))) %>% 
  pivot_longer(cols = everything(), names_to = c(".value", "feature"), names_sep = "_")

# Print the resulting summary data frame
print(summary_df)

# Histograms for numerical variables
dat %>%
  select(-"Outcome") %>%
  gather(key = "feature", value = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 35, fill = "lightpink", colour = "purple") +
  facet_wrap(~feature, scales = "free") + # Creates separate plots for each feature with free axis scales.
  theme_minimal() +
  labs(title = "Features's Distributions after Replace Nonsense Zeros", x = NULL)
# Tall bars due to replacement of nonsense zeros

# Data prepare for creating multiple boxpots
x <- dat %>% select(-"Outcome")
y <- dat %>% 
  pull(Outcome) %>% # extract the 'Outcome' column as a vector
  as.factor()  

# Reshape data to long format and create boxplots
data.frame(y, x) %>%
  gather(key = "feature", value = "value", -y) %>%
  ggplot(aes(feature, value, fill = y)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free") +
  scale_fill_manual(values = c("0" = "lightpink", "1" = "purple")) + # Custom colors
  theme_minimal() +
  labs(title = "Boxplots", x = NULL)
# Median values/distribution of each variable are generally higher in patients with diabetes


# 4.2. Correlation analysis

# Test the correlation between two continuous variables: SkinThickness and Insulin
cor.test(dat$SkinThickness, dat$Insulin)
#  p-value = 0.0000751 indicates that the correlation is statistically significant at the 0.05 level.

# Test the correlation between two continuous variables: Age and BMI
cor.test(dat$Age, dat$BMI)
# p-value = 0.334 indicates that the correlation is not statistically significant at the 0.05 level.

# perform t-test on Glucose
t.test(Glucose ~ as.factor(Outcome), dat) 
# p-value < 2e-16. There is a significant difference between the mean_glucose of two groups '0' and '1'

# Calculate the correlation matrix
cor_matrix <- round(cor(dat),2)

# Create the heatmap with cell values
pheatmap(cor_matrix, 
         main = "Correlation Heatmap with Cell Values",
         color = colorRampPalette(c("white", "lightpink", "purple"))(50),
         display_numbers = TRUE,  # Show numbers in cells
         number_format = "%.2f",  # Format of the numbers
         fontsize_number = 8 # Size of the numbers
)     
# No strong correlations between variables
# Few moderate correlations between variables
# Glucose level has most effective to the outcome
# BloodPressure has least effective to the outcome


######################## 5. Predictive Modeling


# Scale date to ensure that data is uniformly prepared for analysis and modeling
x_scaled <- dat %>% 
  select(-"Outcome") %>% 
  scale()

# Extract the 'Outcome' column as a vector
y <- dat %>% 
  pull(Outcome) %>% 
  as.factor() 

# Seed for reproducibility of data generation
set.seed(1, sample.kind = "Rounding")

# Split data into training and testing sets:

# Ensure that our training and testing sets are representative of the entire dataset
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)

# Split the dataset into training and testing sets
test_x <- x_scaled[test_index,]
test_y <- y[test_index]

train_x <- x_scaled[-test_index,]
train_y <- y[-test_index]

# Train control used for all methods. Specifies 10-fold cross-validation for model evaluation.
train_control <- trainControl(method = "cv", number = 10)


#--------------------------------------
# 5.1. Generalized Linear Model (GLM)
#--------------------------------------

# Train the GLM model
train_glm <- train(train_x, train_y, 
                   method = "glm", 
                   family = binomial,
                   trControl = train_control
                   )

# Predict on the testing set
glm_preds <- predict(train_glm, test_x)

# Create a tibble to store model's accuracy, accuracy
accuracy <- tibble(Model = "Generalized Linear Model (GLM)",
                   Accuracy = round(mean(glm_preds == test_y), 5) # Compare predictions with the actual values
                   )

# Print accuracy tibble
accuracy %>% kable()


#------------------------------------------
# 5.2. Linear Discriminant Analysis (LDA)
#------------------------------------------

# Train the LDA model
train_lda <- train(train_x, train_y, 
                   method = "lda",
                   trControl = train_control
                   )

# Predict on the testing set
lda_preds <- predict(train_lda, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Linear Discriminant Analysis (LDA)",
                         Accuracy = round(mean(lda_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#---------------------------------------------
# 5.3. Quadratic Discriminant Analysis (QDA)
#---------------------------------------------

# Train the QDA model
train_qda <- train(train_x, train_y, 
                   method = "qda",
                   trControl = train_control
                   )

# Predict on the testing set
qda_preds <- predict(train_qda, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Quadratic Discriminant Analysis (QDA)",
                         Accuracy = round(mean(qda_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#------------------------------------------------------
# 5.4. Locally Estimated Scatterplot Smoothing (LOESS)
#------------------------------------------------------

# Train the LOESS model.
train_loess <- train(train_x, train_y, 
                     method = "gamLoess",
                     trControl = train_control
                     )

# Predict on the testing set
loess_preds <- predict(train_loess, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Locally weighted scatterplot smoothing (LOESS)",
                         Accuracy = round(mean(loess_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#---------------------------------
# 5.5. K-Nearest Neighbors (KNN)
#---------------------------------

# Define the tuning grid for specifying a range of hyperparameters or 'neighbors' to tune
tuning <- data.frame(k = seq(3, 21, 2))

# Train the KNN model
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning,
                   trControl = train_control
                   )

# Predict on the testing set
knn_preds <- predict(train_knn, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "K-Nearest Neighbors (KNN)",
                         Accuracy = round(mean(knn_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#---------------------------
# 5.6. Random Forests (RF)
#---------------------------

set.seed(1, sample.kind = "Rounding")

# Create a tuning data frame with different values for mtry. 
# mtry is the number of features to consider for splitting at each node in a decision tree within the Random Forest.
tuning <- data.frame(mtry = c(3, 5, 7, 9))

# Train the RF model.
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  trControl = train_control,
                  importance = TRUE
                  )

# Predict on the testing set
rf_preds <- predict(train_rf, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Random Forests (RF)",
                         Accuracy = round(mean(rf_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()

# Compute variable importance
var_imp <- varImp(train_rf)

# Visualize variable importance
plot(var_imp)
# Glucose, BMI features are most influential in our model.
# Blood pressure has importance score of zero, suggesting it has zero influence in predicting Diabetes


#------------------------
# 5.7. Naive Bayes (NB)
#------------------------

# Train the NB model
train_nb <- train(train_x, y = train_y,
                  method = "naive_bayes",
                  trControl = train_control
                  )

# Predict on the testing set
nb_preds <- predict(train_nb, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Naive Bayes (NB)",
                         Accuracy = round(mean(nb_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#----------------------
# 5.8. Decision Trees
#----------------------

set.seed(1, sample.kind = "Rounding")

# Train the decision tree model
train_dt <- train(x = train_x, y = train_y,
                  method = "rpart",
                  trControl = train_control
                  )

# Predict on the testing set
dt_preds <- predict(train_dt, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Decision Trees (DT)",
                         Accuracy = round(mean(dt_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#-------------------------------------
# 5.9. Support Vector Machines (SVM)
#-------------------------------------
# Method learned from DataCamp tutorial.

set.seed(1, sample.kind = "Rounding")

# Train an SVM model with Radial Basis Function (RBF) Kernel
train_svm <- train(x = train_x, y = train_y,
                   method = "svmRadial",
                   trControl = train_control
                   )

# Predict on the testing set
svm_preds <- predict(train_svm, test_x)

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Support Vector Machines (SVM)",
                         Accuracy = round(mean(svm_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


#----------------------------
# 5. 10. Creating an ensemble
#----------------------------

# Create an ensemble
ensemble <- cbind(glm = glm_preds == 0, 
                  lda = lda_preds == 0, 
                  qda = qda_preds == 0, 
                  loess = loess_preds == 0, 
                  rf = rf_preds == 0, 
                  knn = knn_preds == 0,
                  nb = nb_preds == 0,
                  dt = dt_preds == 0,
                  svm = svm_preds == 0
                  )

# Predict on the testing set. If more models predict "0" then return "0"
ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "0", "1")

# Calculate the accuracy then add on to the accuracy tibble
accuracy <- rbind(accuracy,
                  tibble(Model = "Ensemble",
                         Accuracy = round(mean(ensemble_preds == test_y), 5)
                         )
                  )

# Print accuracy tibble
accuracy %>% kable()


############################### 6. Result

# Model with highest accuracy
print(accuracy[which.max(accuracy$Accuracy),])
