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


# construct a tunable graph learner
gl = 
  mlr_tuning_spaces$keys("classif.*default") %>%
  sapply(function(ts) lts(ts)$get_learner(), USE.NAMES = TRUE) %>%
  {names(.) = stringr::str_remove(names(.), ".default|.rbv2"); .} %>%
  ppl("branch", ., "lrn_") %>% as_learner() %>%
  {.$param_set$values$lrn_branch.selection = to_tune()
   if(.$task_type == "classif") {
     .$param_set$values$classif.svm.type = "C-classification"}
   .} 

# c.f. plot(gl$graph)

params = gl$param_set$ids() %>% {.[grepl("classif", .)]}
for (param in params){
  selection = strsplit(param,"[.]") %>% {paste(.[[1]][1:2], collapse = ".")}
  gl$param_set$add_dep(param, "lrn_branch.selection", CondEqual$new(selection))
}
# applies e.g.
#gl$param_set$add_dep("classif.kknn.k",   "lrn_branch.selection", CondEqual$new("classif.kknn"))
#gl$param_set$add_dep("classif.rpart.cp", "lrn_branch.selection", CondEqual$new("classif.rpart"))

# c.f. for a check
gl$param_set$deps

# add auto_tuner
tgl = gl %>%
  auto_tuner(
    learner    = .,
    measure    = NULL,
    method     = tnr("random_search"),
    resampling = rsmp("cv", folds = 3),
    terminator = trm("evals", n_evals = 10)
  )

# c.f. another check
tgl$learner$param_set$deps
tgl$learner$param_set
#-> deps still there?

# trigger resampling
set.seed(1234)
rr = resample(tsk("iris"), tgl, rsmp("holdout"), store_models = TRUE)

# extract the results
# !!Ich will vermeiden, dass hyperparameter variieren, die gar nicht aktiv sind!!


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
# $ ce                        <dbl> 0.02, 0.05, 0.05, 0.06, 0.07, 0.07, 0.07, 0.30, 0.69, 0.75
# $ selection                 <chr> "svm", "rpart", "rpart", "ranger", "kknn", "kknn", "kknn", "svm", "svm", "xgboost"
# $ glmnet.alpha              <dbl> 0.47, 0.27, 0.65, 0.33, 0.97, 0.29, 0.42, 0.10, 0.79, 0.92
# $ glmnet.s                  <dbl> -6.64, -6.54, 6.16, -8.54, 8.62, 1.21, 5.93, 4.32, 7.04, -9.21
# $ kknn.k                    <dbl> 3.76, 0.64, 3.20, 1.14, 2.76, 1.15, 2.22, 2.78, 2.00, 1.70
# $ kknn.distance             <dbl> 1.01, 1.26, 1.84, 4.37, 3.14, 2.26, 4.51, 2.03, 1.45, 1.39
# $ kknn.kernel               <chr> "gaussian", "rank", "biweight", "cos", "triweight", "epanechnikov", "epanechnikov", "ran…
# $ ranger.mtry.ratio         <dbl> 0.35, 0.44, 0.35, 0.86, 0.82, 0.84, 0.71, 0.66, 0.11, 0.86
# $ ranger.num.trees          <dbl> 354, 1564, 323, 1578, 1436, 1989, 1124, 909, 123, 261
# $ ranger.replace            <lgl> TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE
# $ ranger.sample.fraction    <dbl> 0.48, 0.25, 0.29, 0.24, 0.98, 0.86, 0.37, 0.75, 0.31, 0.83
# $ rpart.cp                  <dbl> -6.01, -8.70, -4.97, -4.98, -2.52, -8.50, -6.41, -5.91, -6.67, -6.85
# $ rpart.minbucket           <dbl> 0.95, 1.99, 2.30, 1.01, 3.56, 4.07, 0.89, 3.24, 2.80, 0.29
# $ rpart.minsplit            <dbl> 4.79, 1.85, 0.96, 1.64, 2.96, 4.23, 3.79, 0.97, 1.37, 2.26
# $ svm.cost                  <dbl> 0.19, -1.05, -5.94, 3.74, -8.27, 9.20, 3.92, -2.11, 0.00, 3.29
# $ svm.degree                <dbl> NA, 3, NA, NA, NA, NA, 3, NA, NA, NA
# $ svm.gamma                 <dbl> NA, -0.36, -0.62, -2.87, 9.07, -6.11, -9.08, 8.65, -8.80, -2.53
# $ svm.kernel                <chr> "linear", "polynomial", "sigmoid", "radial", "sigmoid", "radial", "polynomial", "sigmoid…
# $ xgboost.alpha             <dbl> -3.88, 2.44, -2.14, 5.25, -0.64, 3.12, -3.75, 2.34, -2.59, 4.73
# $ xgboost.colsample_bylevel <dbl> 0.41, 0.14, 0.16, 0.54, 0.23, 0.80, 0.51, 0.73, 0.22, 0.64
# $ xgboost.colsample_bytree  <dbl> 0.70, 0.34, 0.53, 0.17, 0.92, 0.15, 0.47, 0.96, 0.21, 0.41
# $ xgboost.eta               <dbl> -1.13, -5.36, -4.77, -7.04, -6.93, -8.82, -4.91, -3.03, -1.28, -5.75
# $ xgboost.lambda            <dbl> 4.68, -3.87, 2.95, 1.25, 4.76, 0.84, -2.84, 2.75, 3.41, -3.74
# $ xgboost.max_depth         <dbl> 11, 18, 16, 20, 12, 4, 17, 7, 1, 5
# $ xgboost.nrounds           <dbl> 2533, 3059, 1417, 2963, 882, 2432, 3500, 1948, 1083, 3756
# $ xgboost.subsample         <dbl> 0.42, 0.23, 0.61, 0.53, 0.12, 0.91, 0.13, 0.93, 0.90, 0.13
