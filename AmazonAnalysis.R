library(tidyverse)
library(ggmosaic)
library(embed)

setwd("C:/Users/rileyw/AmazonEmployeeAccess")

vroom("train.csv") -> train
vroom("test.csv") -> test

am_recipe <- recipe(ACTION ~ .,data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01, other = "OTHER") %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
  
prep(am_recipe) %>%
  bake(new_data = train)


train



# creating mosaic

ggplot() +
  geom_mosaic(aes(x = product(ROLE_DEPTNAME, ACTION), fill = ACTION))
