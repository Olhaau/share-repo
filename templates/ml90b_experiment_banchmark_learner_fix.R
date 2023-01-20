library(magrittr)
library(dplyr)
library(mlr3)

library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3tuningspaces)

# S. 
#https://stackoverflow.com/questions/65402295/mlr3-using-trafo-transformations-within-a-learner-branch-dependencies-hitting
#https://mlr-org.com/gallery/series/2021-03-11-practical-tuning-series-build-an-automated-machine-learning-system/

source("templates/ml90b_backend.R")

# construct a tunable learner
gl = c("glmnet", "kknn", "ranger", "rpart", "xgboost", "svm") %>% 
  paste0("classif.", .) %>% lrns()  %>%
  ppl("branch", ., "lrn_") %>% as_learner() %>% 
  
  # !! hier ist mein fix 'versteckt'!!
  add_ts_def() 

tgl = gl %>%  auto_tuner(
    learner    = .,
    measure    = NULL,
    method     = tnr("random_search"),
    resampling = rsmp("cv", folds = 3),
    terminator = trm("evals", n_evals = 10)
  ) %>% {.$id = "switch_learner_tuned"; .}

# checks
tgl$learner$param_set$values
plot(tgl$learner$graph)

# trigger resampling
set.seed(1234)
rr = resample(tsk("iris"), tgl, rsmp("holdout"), store_models = TRUE)

# extract the resuls
extract_inner_tuning_archives(rr) %>% 
  select(contains(c("classif", "selection")) & !contains("domain")) %>% 
  relocate("classif.ce", "lrn_branch.selection") %>%
  rename_with(~ stringr::str_remove(.x, "classif.|lrn_branch.")) %>%
  arrange(ce) %>% 
  mutate(
    across(where(is.numeric), ~ round(.x,2)),
    selection = stringr::str_remove(selection, "classif.")
  ) %>% glimpse()

# Rows: 10
# Columns: 26
# $ ce                        <dbl> 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.07, 0.10
# $ selection                 <chr> "xgboost", "xgboost", "kknn", "rpart", "rpart", "xgboost", "ranger", "kknn", "kknn", "sv…
# $ glmnet.alpha              <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA
# $ glmnet.s                  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA
# $ kknn.k                    <dbl> NA, NA, 1.15, NA, NA, NA, NA, 2.76, 2.22, NA
# $ kknn.distance             <dbl> NA, NA, 2, NA, NA, NA, NA, 3, 5, NA
# $ kknn.kernel               <chr> NA, NA, "epanechnikov", NA, NA, NA, NA, "triweight", "epanechnikov", NA
# $ ranger.mtry.ratio         <dbl> NA, NA, NA, NA, NA, NA, 1, NA, NA, NA
# $ ranger.num.trees          <dbl> NA, NA, NA, NA, NA, NA, 1578, NA, NA, NA
# $ ranger.replace            <chr> NA, NA, NA, NA, NA, NA, "TRUE", NA, NA, NA
# $ ranger.sample.fraction    <dbl> NA, NA, NA, NA, NA, NA, 0.24, NA, NA, NA
# $ rpart.cp                  <dbl> NA, NA, NA, -8.70, -4.97, NA, NA, NA, NA, NA
# $ rpart.minbucket           <dbl> NA, NA, NA, 1.99, 2.30, NA, NA, NA, NA, NA
# $ rpart.minsplit            <dbl> NA, NA, NA, 1.85, 0.96, NA, NA, NA, NA, NA
# $ xgboost.alpha             <dbl> 0.14, 0.00, NA, NA, NA, -1.58, NA, NA, NA, NA
# $ xgboost.colsample_bylevel <dbl> 0.38, 0.26, NA, NA, NA, 0.76, NA, NA, NA, NA
# $ xgboost.colsample_bytree  <dbl> 0.23, 0.12, NA, NA, NA, 0.97, NA, NA, NA, NA
# $ xgboost.eta               <dbl> -0.09, -6.24, NA, NA, NA, -3.97, NA, NA, NA, NA
# $ xgboost.lambda            <dbl> -3.88, -2.59, NA, NA, NA, 2.34, NA, NA, NA, NA
# $ xgboost.max_depth         <dbl> 7, 3, NA, NA, NA, 14, NA, NA, NA, NA
# $ xgboost.nrounds           <dbl> 3358, 594, NA, NA, NA, 4769, NA, NA, NA, NA
# $ xgboost.subsample         <dbl> 0.89, 0.88, NA, NA, NA, 0.70, NA, NA, NA, NA
# $ svm.cost                  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, -4.99
# $ svm.degree                <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, 2
# $ svm.gamma                 <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, 4.63
# $ svm.kernel                <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, "polynomial"

