library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(discrim)
library(themis)

cl <- makePSOCKcluster(10)
registerDoParallel(cl)


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

tuning_grid <- grid_regular(mtry(range = c(1, 5)),
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