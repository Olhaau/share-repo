library(magrittr)
library(dplyr)
library(mlr3)

library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3tuningspaces)



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
