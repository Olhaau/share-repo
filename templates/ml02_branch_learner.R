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
extract_inner_tuning_archives(rr) %>% 
  select(contains(c("classif", "selection")) & !contains("domain")) %>% 
  relocate("classif.ce", "lrn_branch.selection") %>%
  rename_with(~ stringr::str_remove(.x, "classif.|lrn_branch.")) %>%
  mutate(
    across(where(is.numeric), ~ round(.x,1)),
    selection = stringr::str_remove(selection, "classif.")
  ) %>% glimpse()


#---

add_ts_def = function(learner, type = "default", branch_name = "lrn_branch"){
  ps_tab = function(key){
    
    vals = lts(key)$values
    paramset = tibble()
    for(val in vals){
      para = val$content
      trafo = para$trafo
      if(!is.null(para$param)){para = para$param}
      
      paramset = bind_rows(
        paramset,
        tibble(
          lower = para$lower,
          upper = para$upper,
          levels = list(para$levels),
          trafo = list(trafo),
          logscale = para$logscale,
        )
      )
    }
    paramset %>% mutate(
      task_type = stringr::str_split(key, "[.]")[[1]][1],
      learner = stringr::str_split(key, "[.]")[[1]][2],
      learner_type = paste0(task_type, ".", learner),
      tuningspace = stringr::str_split(key, "[.]")[[1]][3],
      param = names(vals),
      
      .before = 1)
  }
  
  ps_dict = mlr_tuning_spaces$keys() %>% lapply(ps_tab) %>% bind_rows() %>%
    mutate(logscale = ifelse(is.na(logscale) & !is.na(lower), FALSE, logscale))
  
  
  ps_dep_dict =
    list(
      classif.glmnet = ps_dict %>%
        filter(learner_type == "classif.glmnet" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = list(to_tune(p_dbl(
            lower = lower, upper = upper, logscale = logscale, trafo = trafo,
            depends = lrn_branch.selection == "classif.glmnet"
          )))
        )
      
      ,classif.kknn = ps_dict %>%
        filter(learner_type == "classif.kknn" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = ifelse(
            !is.na(lower) & !is.na(upper),
            ifelse(
              (lower == floor(lower)) & (upper == floor(upper)),
              
              list(to_tune(p_int(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.kknn"
              ))),
              
              list(to_tune(p_dbl(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.kknn"
              )))
              
            ),
            list(to_tune(p_fct(
              levels = unlist(levels),
              depends = lrn_branch.selection == "classif.kknn"
            )))
          )
        )
      
      ,classif.rpart = ps_dict %>%
        filter(learner_type == "classif.rpart" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = ifelse(
            !is.na(lower) & !is.na(upper),
            ifelse(
              (lower == floor(lower)) & (upper == floor(upper)),
              
              list(to_tune(p_int(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.rpart"
              ))),
              
              list(to_tune(p_dbl(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.rpart"
              )))
              
            ),
            list(to_tune(p_fct(
              levels = unlist(levels),
              depends = lrn_branch.selection == "classif.rpart"
            )))
          )
        )
      
      ,classif.ranger = ps_dict %>%
        filter(learner_type == "classif.ranger" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = ifelse(
            !is.na(lower) & !is.na(upper),
            ifelse(
              (lower == floor(lower)) & (upper == floor(upper)),
              
              list(to_tune(p_int(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.ranger"
              ))),
              
              list(to_tune(p_dbl(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.ranger"
              )))
              
            ),
            list(to_tune(p_fct(
              levels = unlist(levels),
              depends = lrn_branch.selection == "classif.ranger"
            )))
          )
        )
      
      ,classif.xgboost = ps_dict %>%
        filter(learner_type == "classif.xgboost" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = ifelse(
            !is.na(lower) & !is.na(upper),
            ifelse(
              (lower == floor(lower)) & (upper == floor(upper)),

              list(to_tune(p_int(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.xgboost"
              ))),

              list(to_tune(p_dbl(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.xgboost"
              )))

            ),
            list(to_tune(p_fct(
              levels = unlist(levels),
              depends = lrn_branch.selection == "classif.xgboost"
            )))
          )
        )
      
      ,classif.svm = ps_dict %>%
        filter(learner_type == "classif.svm" ) %>%
        select(-c(task_type, learner)) %>%
        rowwise() %>%
        mutate(
          val_dep = ifelse(
            !is.na(lower) & !is.na(upper),
            ifelse(
              (lower == floor(lower)) & (upper == floor(upper)),
              
              list(to_tune(p_int(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.svm"
              ))),
              
              list(to_tune(p_dbl(
                lower = lower, upper = upper, logscale = logscale,
                depends = lrn_branch.selection == "classif.svm"
              )))
              
            ),
            list(to_tune(p_fct(
              levels = unlist(levels),
              depends = lrn_branch.selection == "classif.svm"
            )))
          )
        )
      
      
      
      #classif: xgboost #-> errors
      #regr: glmnet, kknn, ranger, rpart, svm, xgboost
      
    ) %>% bind_rows()
  
  #' Set Parameter of an mlr3-learner.
  #'
  #' @param obj: An (mlr3)-learner
  #' @param params: Named list of parameter to change, delete.
  #' @return The Object [obj] with modified params.
  set_param = function(learner, params = list()){
    for(param in names(params)){
      if(param %in% learner$param_set$ids()){
        learner$param_set$values[[param]] = params[[param]]
      }
    }
    learner
  }
  
  if(paste0(learner$id, ".", type) %in% mlr_tuning_spaces$keys()){
    lts(paste0(learner$id, ".", type))$get_learner() %>%
      {set_param(., list(id = paste0(.$id, ".", stringr::str_sub(type, end = 3L))))}
  } else {
    meths = learner$param_set$params[[paste0(branch_name, ".selection")]]$levels
    
    params = ps_dep_dict %>%
      filter(learner_type %in% meths, tuningspace == "default") %>%
      mutate(param = paste0(learner_type, ".", param)) %>%
      rename(values = val_dep) %>%
      select(param, values)
    
    pn = params$param
    params = params$values
    names(params) = pn
    
    params[[paste0(branch_name, ".selection")]] = to_tune()
    learner %>% set_param(params) %>%
      {set_param(., list(id = paste0(.$id, ".", stringr::str_sub(type, end = 3L))))} 
  }
  if("classif.svm" %in% learner$param_set$params$lrn_branch.selection$levels){}
  learner = learner %>% set_param(list(classif.svm.type = "C-classification"))
}

tgl = c("glmnet", "kknn", "ranger", "rpart", "xgboost", "svm"
) %>% paste0("classif.", .) %>% lrns()  %>%
  mlr3pipelines::ppl("branch", ., "lrn_") %>% 
  as_learner() %>% add_ts_def() %>%
  auto_tuner(
    learner    = .,
    measure    = NULL,
    method     = tnr("random_search"),
    resampling = rsmp("cv", folds = 3),
    terminator = trm("evals", n_evals = 10)
  ) %>% {.$id = "switch_learner_tuned"; .}

# check
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
