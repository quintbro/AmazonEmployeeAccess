#--------------- Recipe and Read in Data --------------------

library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
# library(doParallel)
library(discrim)
library(themis)

vroom("train.csv") -> train
vroom("test.csv") -> test

train %>%
  mutate(ACTION = as_factor(ACTION)) -> train

am_recipe <- recipe(ACTION ~ .,data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .9) %>%
  step_smote(all_outcomes(), neighbors = 5)

# -------------- Logistic Regression -------------------

log_mod <- logistic_reg() %>%
  set_engine("glm")

log_wf <- workflow() %>%
  add_model(log_mod) %>%
  add_recipe(am_recipe) %>%
  fit(data = train)

preds <- predict(log_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "logistic.csv", delim = ",")

# ---------------- Penalized Regression -----------------

pen_mod <- logistic_reg(mixture = tune(),
                        penalty = tune()) %>%
  set_engine("glmnet")

pen_wf <- workflow() %>%
  add_recipe(am_recipe) %>%
  add_model(pen_mod)

tuning_grid <- grid_regular(
  mixture(),
  penalty(),
  levels = 20
)

folds <- vfold_cv(data = train, v = 5, repeats = 1)

cv_results <- pen_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

params <- cv_results %>%
  select_best("roc_auc")

final_pen_wf <- pen_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_pen_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "penalized.csv", delim = ",")

#------------------ Classification Random Forests ------------------

crf_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

crf_wf <- workflow() %>%
  add_model(crf_mod) %>%
  add_recipe(am_recipe)

tuning_grid <- grid_regular(mtry(range = c(1, 6)),
                            min_n(),
                            levels = 20)

folds <- vfold_cv(train, v = 10)

cv_results <- crf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

params <- cv_results %>%
  select_best("roc_auc")

print(params) # mtry: 4, min_n: 4

final_crf_wf <- crf_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_crf_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "randomforest.csv", delim = ",")

#-------------------- Naive Bayes -------------------------

nb_mod <- naive_Bayes(smoothness = tune(),
                      Laplace = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_model(nb_mod) %>%
  add_recipe(am_recipe)

cv_grid <- grid_regular(smoothness(),
                        Laplace(),
                        levels = 10)
folds <- vfold_cv(train, v = 10)

cv_results <- nb_wf %>%
  tune_grid(grid = cv_grid,
            resamples = folds,
            metrics = metric_set(roc_auc))

params <- cv_results %>%
  select_best("roc_auc")

final_nb_wf <- nb_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_nb_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "naivebayes.csv", delim = ",")

#----------------- K Nearest Neighbors --------------------

knn_mod <- nearest_neighbor(neighbors = tune(),
                            dist_power = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <- workflow() %>%
  add_model(knn_mod) %>%
  add_recipe(am_recipe)

param_grid <- grid_regular(neighbors(), # neighbors = 10, dist_power() = 1
                           dist_power(),
                           levels = 10)
folds <- vfold_cv(train, v = 10)

cv_results <- knn_wf %>%
  tune_grid(grid = param_grid,
            resamples = folds,
            metrics = metric_set(roc_auc))

params <- cv_results %>%
  select_best("roc_auc")

print(params)

final_knn_wf <- knn_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_knn_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "knn.csv", delim = ",")


#------------------ Support Vector Machines ----------------------

svm_mod <- svm_rbf(rbf_sigma = tune(),
                   cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(am_recipe)

param_grid <- grid_regular(rbf_sigma(),
                           cost(),
                           levels = 10)

folds <- vfold_cv(train, v = 5)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(roc_auc))

params <- cv_results %>%
  select_best("roc_auc")

print(params)

final_svm_wf <- svm_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_svm_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "svm.csv", delim = ",")
