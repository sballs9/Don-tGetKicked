library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)
library(ranger)
library(stacks)
library(glmnet)

train <- vroom::vroom("training.csv",
                      na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom::vroom("test.csv",
                     na = c("", "NA", "NULL", "NOT AVAIL"))

train <- train %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

# Recipe

my_recipe <- recipe(IsBadBuy ~., data = train) %>%
  update_role(RefId, new_role = 'ID') %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate,
          AUCGUART, PRIMEUNIT,
          Model, SubModel, Trim) %>% 
  step_corr(all_numeric_predictors(), threshold = .8) %>%
  step_other(all_nominal_predictors(), threshold = .005) %>% #Originally 0.0001
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_knn(all_numeric_predictors())

# Cross Validation

folds <- vfold_cv(train, v = 5, repeats=1)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples() 

# Boosting

boosted_model <- boost_tree(tree_depth=4,
                            trees=2000,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model) %>%
  fit(data=train)

boost_models <- fit_resamples(boost_wf,
                              resamples = folds,
                              metrics = metric_set(roc_auc),
                              control = tunedModel)

# Random forest
classForest_model <- rand_forest(mtry = tune(), # how many var are considered
                                 min_n=tune(), # how many observations per leaf
                                 trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

classForest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(classForest_model)

forest_tuning_grid <- grid_regular(mtry(range =c(1,1)),
                                   min_n(),
                                   levels = 5)

forest_models <- classForest_wf %>%
  tune_grid(resamples=folds,
            grid=forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

# BART

bart_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>% # address novel factor levels in categorical (nominal) predictors that might appear in new data but are not present in the training set. 
  step_unknown(all_nominal_predictors()) %>% # Creates a new factor level for 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .8) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

bart_mod <- parsnip::bart(trees = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_mod)

bart_tuning_grid <- grid_regular(trees(),
                                 levels = 5)

bart_models <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tuning_grid,
            metrics = metric_set(roc_auc),
            control = untunedModel)

# Stacked Model

my_stack <- stacks() %>%
  add_candidates(forest_models) %>%
  add_candidates(boost_models) %>%
  add_candidates(bart_models)

stack_mod <- my_stack %>%
  blend_predictions() %>% 
  fit_members()

# Predictions 

stack_preds <- stack_mod %>%
  predict(new_data=test,type="prob") %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)

vroom_write(x=stack_preds, file="./submission.csv", delim=",")