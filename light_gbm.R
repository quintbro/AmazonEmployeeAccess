library(tidymodels)
library(tidyverse)
library(vroom)
library(doParallel)
library(bonsai)
setwd("C:/Users/rileyw/AmazonEmployeeAccess")
vroom("train.csv") -> train
vroom("test.csv") -> test

cl = makePSOCKcluster(5)
registerDoParallel(cl)

train %>%
  mutate(ACTION = as_factor(ACTION)) -> train

am_recipe <- recipe(ACTION ~ .,data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor)

lgb_mod <- boost_tree(
  # mtry = 5,
  # min_n = 10,
  # tree_depth = 10,
  learn_rate = .04,
  trees = 500,
  loss_reduction = 2.5
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

lgb_wf <- workflow() %>%
  add_recipe(am_recipe) %>%
  add_model(lgb_mod) %>%
  fit(train)

# gridTune <- grid_regular(mtry(range = c(1, 8)),
#                          min_n(),
#                          tree_depth(),
#                          levels = 5)
# folds <- vfold_cv(train, v = 5)
# 
# cv_results <- lgb_wf %>%
#   tune_grid(grid = gridTune,
#             resamples = folds,
#             metrics = metric_set(roc_auc))
# 
# params <- cv_results %>% select_best("roc_auc")
# 
# final_wf <- lgb_wf %>%
#   finalize_workflow(params) %>%
#   fit(train)

# final_wf <- lgb_wf %>%
#   fit(train)

preds <- predict(lgb_wf, new_data = test, type = "prob")

# stopCluster(cl)

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "lightgbm.csv", delim = ",")
