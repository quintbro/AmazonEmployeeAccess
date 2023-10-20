# library(tidyverse)
# library(tidymodels)
# # library(ggmosaic)
# library(embed)
# library(vroom)
# 
# # setwd("C:/Users/rileyw/AmazonEmployeeAccess")
# 
# vroom("train.csv") -> train
# vroom("test.csv") -> test
# 
# am_recipe <- recipe(ACTION ~ .,data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .01, other = "OTHER") %>%
#   step_dummy(all_nominal_predictors())
# 
# prep(am_recipe) %>%
#   bake(new_data = train)
# 
# 
# train %>%
#   select(ACTION, ROLE_TITLE) %>%
#   table()
# 
# # creating mosaic
# #train %>%
# #  mutate_all(as_factor) %>%
# #ggplot() +
# #  geom_mosaic(aes(x = product(ACTION, ROLE_FAMILY), fill = ACTION))
# 
# 
# length(unique(train$RESOURCE))
# length(unique(train$MGR_ID))
# length(unique(train$ROLE_ROLLUP_1))
# length(unique(train$ROLE_ROLLUP_2))
# length(unique(train$ROLE_DEPTNAME))
# length(unique(train$ROLE_TITLE))
# length(unique(train$ROLE_FAMILY_DESC))
# length(unique(train$ROLE_FAMILY))
# length(unique(train$ROLE_CODE))
# 
# 
# #----------- Logistic Regression --------------------
# library(tidyverse)
# library(tidymodels)
# library(vroom)
# 
# 
# # setwd("C:/Users/rileyw/AmazonEmployeeAccess")
# 
# vroom("train.csv") -> train
# vroom("test.csv") -> test
# 
# train %>%
#   mutate(ACTION = as_factor(ACTION)) -> train
# 
# am_recipe <- recipe(ACTION ~ .,data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .01, other = "OTHER")%>%
#   step_dummy(all_nominal_predictors())
# 
# prep(am_recipe) %>%
#   bake(new_data = train)
# 
# log_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# log_wf <- workflow() %>%
#   add_model(log_mod) %>%
#   add_recipe(am_recipe) %>%
#   fit(data = train)
# 
# preds <- predict(log_wf, new_data = test, type = "prob")
# 
# preds %>%
#   mutate(Id = test$id, Action = .pred_1) %>%
#   select(Id, Action) %>%
#   vroom_write(., "submission.csv", delim = ",")
# 
# 
# #------------- Penalized Logistic Regression ------------------
# 
# # library(tidyverse)
# # library(tidymodels)
# # library(vroom)
# # library(embed)
# 
# 
# # setwd("C:/Users/rileyw/AmazonEmployeeAccess")
# 
# vroom("train.csv") -> train
# vroom("test.csv") -> test
# 
# train %>%
#   mutate(ACTION = as_factor(ACTION)) -> train
# 
# am_recipe <- recipe(ACTION ~ .,data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001, other = "OTHER")%>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# pen_mod <- logistic_reg(mixture = tune(),
#                         penalty = tune()) %>%
#   set_engine("glmnet")
# 
# pen_wf <- workflow() %>%
#   add_recipe(am_recipe) %>%
#   add_model(pen_mod)
# 
# tuning_grid <- grid_regular(
#   mixture(),
#   penalty(),
#   levels = 20
# )
# 
# folds <- vfold_cv(data = train, v = 5, repeats = 1)
# 
# cv_results <- pen_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# params <- cv_results %>%
#   select_best("roc_auc")
# 
# final_wf <- pen_wf %>%
#   finalize_workflow(params) %>%
#   fit(data = train)
# 
# preds <- predict(final_wf, new_data = test, type = "prob")
# 
# preds %>%
#   mutate(Id = test$id, Action = .pred_1) %>%
#   select(Id, Action) %>%
#   vroom_write(., "submission.csv", delim = ",")

# #--------------- Classification Random Forests --------------------

# library(tidymodels)
# library(tidyverse)
# library(vroom)
# library(embed)
# library(doParallel)
# 
# cl <- makePSOCKcluster(30)
# registerDoParallel(cl)
# 
# # setwd("C:/Users/rileyw/AmazonEmployeeAccess")
# 
# vroom("train.csv") -> train
# vroom("test.csv") -> test
# 
# train %>%
#   mutate(ACTION = as_factor(ACTION)) -> train
# 
# am_recipe <- recipe(ACTION ~ .,data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001, other = "OTHER")%>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# crf_mod <- rand_forest(
#   mtry = tune(),
#   min_n = tune(),
#   trees = 1000
# ) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# crf_wf <- workflow() %>%
#   add_model(crf_mod) %>%
#   add_recipe(am_recipe)
# 
# tuning_grid <- grid_regular(mtry(range = c(1, 9)),
#                             min_n(),
#                             levels = 20)
# 
# folds <- vfold_cv(train, v = 10)
# 
# cv_results <- crf_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# params <- cv_results %>%
#   select_best("roc_auc")
# 
# print(params) # mtry: 4, min_n: 4
# 
# final_wf <- crf_wf %>%
#   finalize_workflow(params) %>%
#   fit(data = train)
# 
# preds <- predict(final_wf, new_data = test, type = "prob")
# 
# stopCluster(cl)
# 
# preds %>%
#   mutate(Id = test$id, Action = .pred_1) %>%
#   select(Id, Action) %>%
#   vroom_write(., "submission.csv", delim = ",")


# #------------------------- Naive Bayes ---------------------------
# 
# library(tidyverse)
# library(tidymodels)
# library(embed)
# library(vroom)
# library(discrim)
# library(doParallel)
# 
# # setwd("C:/Users/rileyw/AmazonEmployeeAccess")
# 
# vroom("train.csv") -> train
# vroom("test.csv") -> test
# 
# train %>%
#   mutate(ACTION = as_factor(ACTION)) -> train
# 
# am_recipe <- recipe(ACTION ~ .,data = train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001, other = "OTHER")%>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# nb_mod <- naive_Bayes(smoothness = tune(),
#                       Laplace = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes") 
# 
# nb_wf <- workflow() %>%
#   add_model(nb_mod) %>%
#   add_recipe(am_recipe)
# 
# cv_grid <- grid_regular(smoothness(),
#                         Laplace(),
#                         levels = 10)
# folds <- vfold_cv(train, v = 10)
# 
# cl = makePSOCKcluster(10)
# registerDoParallel(cl)
# 
# cv_results <- nb_wf %>%
#   tune_grid(grid = cv_grid,
#             resamples = folds,
#             metrics = metric_set(roc_auc))
# 
# params <- cv_results %>%
#   select_best("roc_auc")
# 
# final_wf <- nb_wf %>%
#   finalize_workflow(params) %>%
#   fit(data = train)
# 
# preds <- predict(final_wf, new_data = test, type = "prob")
# 
# stopCluster(cl)
# 
# preds %>%
#   mutate(Id = test$id, Action = .pred_1) %>%
#   select(Id, Action) %>%
#   vroom_write(., "submission.csv", delim = ",")


# #--------- K-Nearest Neighbors -----------

library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)


# setwd("C:/Users/rileyw/AmazonEmployeeAccess")

vroom("train.csv") -> train
vroom("test.csv") -> test

train %>%
  mutate(ACTION = as_factor(ACTION)) -> train

am_recipe <- recipe(ACTION ~ .,data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001, other = "OTHER")%>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

knn_mod <- nearest_neighbor(neighbors = tune(),
                            dist_power = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <- workflow() %>%
  add_model(knn_mod) %>%
  add_recipe(am_recipe)

param_grid <- grid_regular(neighbors(),
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

final_wf <- knn_wf %>%
  finalize_workflow(params) %>%
  fit(data = train)

preds <- predict(final_wf, new_data = test, type = "prob")

preds %>%
  mutate(Id = test$id, Action = .pred_1) %>%
  select(Id, Action) %>%
  vroom_write(., "submission.csv", delim = ",")