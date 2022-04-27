library(dplyr)
library(naniar)
library(splitstackshape)
library(catboost)
set.seed(123)

catboost_prepare_data <- function(data, target, continuous_names = NULL, drops = NULL, char_na = NULL){
  'This function prepares given data in order to preform catboost modeling on it.
  
   It returns the list of data frames, which are:
    X_train - all independent variables from the sample of 70% of given data , 
              will be used to train the model
    X_test - the rest (30%) of independent variables, 
             which will be used to make predictions to test the accuracy of the model
    y_train - 70% of the records of dependent variable, corresponding to X_train 
    y_test - records of dependent variable, corresponding to X_test, 
            which will be used to test the accuracy between actual values and predictions
  
   Arguments:
    data - the data frame to prepare
    target - targeted variable of the model
    continuous_names - a vector of names of continuous columns in data frame
    drops - a vector of names of highly correlated columns to drop,
            based on correlation matrix
    char_na - character or double used to substitute NA in the data frame'
  
  ## Data cleaning - removing data with nearly all variables blank 
  data <- data %>% mutate_all(funs(replace(., .== char_na, NA))) %>%
    na.omit()
  
  ## Encoding continuous variables by grouping
  
#   data[continuous_names] <- data %>%select(continuous_names) %>% sapply(function(x){
#     q <- quantile(x, probs = c(0.33,0.66,1))
#     x <- cut(x, breaks = c(-Inf,q[1],q[2],q[3]), 
#              labels = c(0,1,2))
#   })
  
  ## Encoding character data with integer labels
  data[, sapply(data, class) == 'character']<-  data %>% select(where(is.character)) %>%  
    sapply(function(x) as.integer(as.factor(x)) )
  
  ## Remove highly correlated variables
  data <- data[ , !(names(data) %in% drops)]
  
  ## Stratifying the data
  
  all <- stratified(data, c(target), 0.7, bothSets = TRUE)
  X_train <- all$SAMP1
  X_test <- all$SAMP2
  y_train <- X_train %>%
    pull(target) %>%
    as.numeric()
  print(y_train)
  y_test <- X_test %>%
    pull(target) %>%
    as.numeric()
  exclude <- c(target)
  X_train <- select(X_train, -exclude) %>% sapply(function(x){as.numeric(x)})
  X_test <- select(X_test, -exclude) %>% sapply(function(x){as.numeric(x)})
  
  list <- list("X_train" = X_train, "X_test" = X_test, "y_train" = y_train, "y_test" = y_test)
  
  return(list)
}

catboost_function <- function(X_train,y_train, params=list(loss_function = 'RMSE')){
  'This function builds the catboost model with default parameters.
   The outcome of the function is mentioned model.
  
   Arguments:
    X_train - all independent variables from the sample of 70% of given data , 
              will be used to train the model, should be prepared by the function 
              `catboost_prepare_data`
    y_train - 70% of the records of dependent variable, corresponding to X_train,
              should be prepared by the function `catboost_prepare_data`
    params - list of parameters to base the model on, as for this version: list(loss_function = "RMSE") by changing 
              loss_funtion you can choose what tipe of problem it is going to be solved:
              for example (regression: "RMSE", binar classification: "Logloss", multiclassification: "MultiClass")'
  
  ## Build the model
  train_pool <- catboost.load_pool(data = X_train, label = y_train)
  model <- catboost.train(train_pool, params = params)
  
  return(model)
}

catboost_predict <- function(model, X_test, y_test = NULL, type = 'RawFormulaVal'){
  'This function returns the predicitons of given data set, based on the used model
  
   Arguments:
    model - desired catboost model built with `catboost_function`
    X_test - the rest (30%) of independent variables, 
             which will be used to make predictions to test the accuracy of the model,
             should have been prepared with `catboost_prepare_data`
    y_test - y_test - records of dependent variable, corresponding to X_test, 
            which will be used to test the accuracy between actual values and predictions,
            should have been prepared with `catboost_prepare_data`
    type - for classification and multiclassification use "Class"
  '

  
  ## Predict the results from X_test
  
  test_pool <- catboost.load_pool(data = X_test, label = y_test)
  
  predict <- catboost.predict(model, 
                              test_pool,
                              prediction_type = type)
  
  return(as.vector(predict))
}

## Example of use - regression:

### Loading the data and necessary variables
bangalore <- read.csv('Bangalore.csv') # 'Price' will be the target variable
drops <- c("WashingMachine","BED","Microwave","TV","DiningTable","Sofa","Wardrobe",
           "Refrigerator","Wifi")
char_na <- 9

### Preparing the data
prepared_data <- catboost_prepare_data(data = bangalore,target = "Price",continuous_names = NULL, drops = drops, char_na =char_na)

### Assigning the variables
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

### Building the model
catboost_model <- catboost_function(X_train,y_train, params = list(loss_function='RMSE'))

### Making the predictions
predict <- catboost_predict(catboost_model,X_test)

### Metrics
#### MSE on predition
sqrt(sum((predict-y_test)^2)/length(predict))
#### MSE on mean
sqrt(sum((mean(y_test)-y_test)^2)/length(predict))

## Example of use - muliclassification:

### Loading the data and necessary variables
bangalore <- read.csv('Bangalore.csv') # 'Price' will be the target variable
drops <- c("WashingMachine","BED","Microwave","TV","DiningTable","Sofa","Wardrobe",
           "Refrigerator","Wifi")
continuous_names <- c('Area','Price')
char_na <- 9

### Preparing the data
prepared_data <- catboost_prepare_data(data = bangalore,target = "Price",continuous_names = continuous_names, drops = drops, char_na =char_na)

### Assigning the variables
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

### Building the model
catboost_model <- catboost_function(X_train,y_train, params = list(loss_function='MultiClass'))

### Making the predictions
predict <- catboost_predict(catboost_model,X_test, type = 'Class')
sum(predict == (y_test-1))/length(predict)



