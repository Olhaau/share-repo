library(magrittr)
library(dplyr)
library(ggplot2)

library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3tuningspaces)
library(mlr3hyperband)



source("templates/ml90b_backend.R")

# construct a tunable learner
gl = c("glmnet", "kknn", "ranger", "rpart", "xgboost", "svm") %>% 
  paste0("classif.", .) %>% lrns()  %>%
  ppl("branch", ., "lrn_") %>% as_learner() %>% 
  # !! hier ist mein fix 'versteckt'!!
  add_ts_def() 


htgl = gl %>% 
  #budget
  {as_learner(
    po("subsample", frac = to_tune(p_dbl(2^-3, 1, tags = "budget"))) %>>% 
      .$graph)
  } %>% 
  # stability
  {.$encapsulate = c(train = "evaluate", predict = "evaluate")
   .$timeout = c(train = 30, predict = 30)
   .$fallback = lrn("classif.featureless")
   .} %>%
  # tuning
  auto_tuner(
    method = tnr("hyperband", eta = 2),
    resampling = rsmp("cv", folds = 3)
  ) %>% {.$id = "hyper_learner"; .}

# trigger
task = tsk("penguins") %>% {ppl("robustify", task = .)$train(.)[[1]]}
set.seed(1234)
rr = resample(task, htgl, rsmp("holdout"), store_models = TRUE)

# extract
extract_inner_tuning_archives(rr) %>% 
  select(contains(c("classif", "selection", "subsample")) & !contains("domain")) %>% 
  rename_with(~ stringr::str_remove(.x, "classif.|lrn_branch.")) %>%
  mutate(
    across(where(is.numeric), ~ round(.x,4)),
    selection = stringr::str_remove(selection, "classif.")
  ) %>% select("ce", "selection", "subsample.frac") %>% 
  arrange(desc(subsample.frac), ce)

#         ce selection subsample.frac
#  1: 0.0174    ranger          1.000
#  2: 0.0174       svm          1.000
#  3: 0.0261       svm          1.000
#  4: 0.0438    glmnet          1.000
#  5: 0.1789     rpart          1.000
#  6: 0.4589   xgboost          1.000
#  7: 0.5415       svm          1.000
#  8: 0.7208   xgboost          1.000
#  9: 0.0218       svm          0.500
# 10: 0.0261       svm          0.500
# 11: 0.0262      kknn          0.500
# 12: 0.0392    ranger          0.500
# 13: 0.0438    glmnet          0.500
# 14: 0.0656     rpart          0.500
# 15: 0.5415       svm          0.500
# 16: 0.6507   xgboost          0.500
# 17: 0.6770   xgboost          0.500
# 18: 0.0305    ranger          0.250
# 19: 0.0392    glmnet          0.250
# 20: 0.0521       svm          0.250
# 21: 0.0566      kknn          0.250
# 22: 0.0784     rpart          0.250
# 23: 0.0874      kknn          0.250
# 24: 0.1135     rpart          0.250
# 25: 0.2229    ranger          0.250
# 26: 0.2232    ranger          0.250
# 27: 0.2389       svm          0.250
# 28: 0.0263    glmnet          0.125
# 29: 0.3184     rpart          0.125
# 30: 0.3619     rpart          0.125
# 31: 0.4933       svm          0.125
# 32: 0.5415      kknn          0.125
# 33: 0.5415     rpart          0.125
# 34: 0.5985     rpart          0.125
# 35: 0.7024     rpart          0.125

extract_inner_tuning_archives(rr) %>% 
  select(contains(c("classif","selection","subsample")) &!contains("domain")) %>% 
  rename_with(~ stringr::str_remove(.x, "classif.|lrn_branch.")) %>%
  mutate(
    selection = stringr::str_remove(selection, "classif."),
    subsample.frac = as.factor(subsample.frac)
    ) %>%
  ggplot(aes(x = subsample.frac, y = ce, color = selection)) + geom_boxplot()

