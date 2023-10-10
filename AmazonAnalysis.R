library(tidyverse)
library(ggmosaic)
library(embed)
library(vroom)

setwd("C:/Users/rileyw/AmazonEmployeeAccess")

vroom("train.csv") -> train
vroom("test.csv") -> test

am_recipe <- recipe(ACTION ~ .,data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01, other = "OTHER")%>%
  step_dummy(all_nominal_predictors()) # %>%


  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
  
prep(am_recipe) %>%
  bake(new_data = train)


train %>%
  select(ACTION, ROLE_TITLE) %>%
  table()

# creating mosaic
train %>%
  mutate_all(as_factor) %>%
ggplot() +
  geom_mosaic(aes(x = product(ACTION, ROLE_FAMILY), fill = ACTION))


length(unique(train$RESOURCE))
length(unique(train$MGR_ID))
length(unique(train$ROLE_ROLLUP_1))
length(unique(train$ROLE_ROLLUP_2))
length(unique(train$ROLE_DEPTNAME))
length(unique(train$ROLE_TITLE))
length(unique(train$ROLE_FAMILY_DESC))
length(unique(train$ROLE_FAMILY))
length(unique(train$ROLE_CODE))
