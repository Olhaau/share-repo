# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

SEED = 1234

# -> !!!! param_sets und learner trennen (sonst gibt es Schwierigkeiten mit deps) !!!!
# S.


# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # Load other packages as needed. # nolint
library(magrittr)

# Set target options:
tar_option_set(
  packages = c(
    "tibble",
    "autoR",
    "dplyr",
    "mlr3",
    "mlr3learners",
    "mlr3tuningspaces",
    "mlr3pipelines",
    "magrittr",
    "mlr3hyperband",
    "paradox"
  ), # packages that your targets need to run
  memory = "transient",
  garbage_collection = TRUE,
  format = "qs" # default storage format
  # Set other options as needed.
)


# Set mlr3 options ------------------------------------------------------------#
# set mlr3 options globally: suppress progress output of `benchmark()`
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")
# S. https://github.com/mlr-org/mlr3-targets
# tar_make_clustermq() configuration (okay to leave alone):
#options(clustermq.scheduler = "multicore")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.
# Run the R scripts in the R/ folder with your custom functions:
#tar_source()
# source("other_functions.R") # Source other scripts as needed. # nolint

list(
  # input ----------------------------------------------------------------------
  tar_target(data, c(
    mlr_tasks %>%
      as.data.table() %>% filter(task_type == "classif") %>%{.$key}
    #,mlr_tasks %>%
    #  as.data.table() %>%filter(task_type == "regr") %>%{.$key}
  ))
  
  ,tar_target(tasks_raw, tsk(data), pattern = data, iteration = "list")
  
  # preprocess -----------------------------------------------------------------
  ,tar_target(tasks, ppl("robustify", task = tasks_raw)$train(tasks_raw)[[1]],
              pattern = map(tasks_raw), iteration = "list")
  
  
  # ml setup --------------------------------------------------------------
  
  # FIXME: into package
  
  ,tar_target(ps_dep_dict, {
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
        
        # ,classif.xgboost = ps_dict %>%
        #   filter(learner_type == "classif.xgboost" ) %>%
        #   select(-c(task_type, learner)) %>%
        #   rowwise() %>%
        #   mutate(
        #     val_dep = ifelse(
        #       !is.na(lower) & !is.na(upper),
        #       ifelse(
        #         (lower == floor(lower)) & (upper == floor(upper)),
        #
        #         list(to_tune(p_int(
        #           lower = lower, upper = upper, logscale = logscale,
        #           depends = lrn_branch.selection == "classif.xgboost"
        #         ))),
        #
        #         list(to_tune(p_dbl(
        #           lower = lower, upper = upper, logscale = logscale,
        #           depends = lrn_branch.selection == "classif.xgboost"
        #         )))
        #
        #       ),
        #       list(to_tune(p_fct(
        #         levels = unlist(levels),
        #         depends = lrn_branch.selection == "classif.xgboost"
        #       )))
        #     )
        #   )
        
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
    
    ps_dep_dict
  })
  ,tar_target(add_ts_def, function(learner, type = "default", branch_name = "lrn_branch"){
    if(paste0(learner$id, ".", type) %in% mlr_tuning_spaces$keys()){
      lts(paste0(learner$id, ".", type))$get_learner() %>%
        {autoR::set_param(., list(id = paste0(.$id, ".", stringr::str_sub(type, end = 3L))))}
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
      learner %>% autoR::set_param(params) %>%
        {autoR::set_param(., list(id = paste0(.$id, ".", stringr::str_sub(type, end = 3L))))}
    }
  })
  
  
  ,tar_target(resamplings, list(cv3 = rsmp("cv", folds = 3, id = "cv3")))
  
  
  ,tar_target(baselearners, list(
    lrn("classif.featureless", id = "fl")
    ,lrn("classif.naive_bayes", id = "nb")
    ,lrn("classif.rpart", id = "rpart")
    ,lrn("classif.ranger", id = "ranger")
  ))
  
  ,tar_target(learners, list(
    
    "kknn" %>% paste0("classif.",.) %>% lrn()#set_param(list(k = to_tune(1,10))) %>%
    ,"rpart" %>% paste0("classif.",.) %>% lrn()
    ,"ranger" %>% paste0("classif.",.) %>%lrn()
    
    ,c("glmnet", "kknn", "ranger", "rpart"
    ) %>% paste0("classif.", .) %>% lrns()  %>%
      mlr3pipelines::ppl("branch", ., "lrn_")
    %>% as_learner() %>% set_prop(list(id = "branch_fast"))
    
    ,c("glmnet", "kknn", "ranger", "rpart", "xgboost"
    ) %>% paste0("classif.", .) %>% lrns()  %>%
      mlr3pipelines::ppl("branch", ., "lrn_")
    %>% as_learner() %>% set_prop(list(id = "branch_mid"))
    
    ,c("glmnet", "kknn", "ranger", "rpart", "xgboost", "svm"
    ) %>% paste0("classif.", .) %>% lrns()  %>%
      mlr3pipelines::ppl("branch", ., "lrn_")
    %>% as_learner() %>% set_prop(list(id = "branch_full"))
    
    # FIXME: add featureselection...
    # FiXME: Stacking, s. https://mlr-org.com/gallery/pipelines/2020-04-27-tuning-stacking/
  ))
  
  ,tar_target(ts_types, list("default", "rbv2"))
  
  
  ,tar_target(tuneablelearners, {
    add_ts_def(learners[[1]], type = ts_types[[1]])},
    pattern = cross(learners, ts_types), iteration = "list")
  
  ,tar_target(tuners, list(
    
    rs10   = function(learner) tuner_def(learner, id = "rs10")
    ,rs10k1 = function(learner) tuner_def(learner, trm_k = 1, id = "rs10k1")
    ,rs25   = function(learner) tuner_def(learner, trm_evals = 25, id = "rs25")
    ,rs10k3 = function(learner) tuner_def(learner, trm_k = 3, id = "rs10k3")
    
    
    ,hb2    = function(learner){
      
      if(!is.null(learner$graph)){learner = learner$graph}
      
      learner = as_learner(
        po("subsample", frac =  to_tune(p_dbl(2^-2, 1, tags = "budget"))) %>>%
          learner)
      
      learner$fallback = lrn("classif.featureless")
      
      at = auto_tuner(
        learner = learner,
        method = "hyperband",
        eta = 2,
        resampling = rsmp("holdout", ratio = 2/3)
      )
      at$id = paste0(at$id, ".hb2")
      at
    }
    
    ,rs25k2   = function(learner) tuner_def(learner, trm_evals = 25,
                                            trm_k = 2, id = "rs25k2")
    
    ,rs50k1   = function(learner) tuner_def(learner, trm_evals = 50,
                                            trm_k = 1, id = "rs50k1")
    
    ,rs10k10   = function(learner) tuner_def(learner, trm_evals = 10,
                                             trm_k = 10, id = "rs10k10")
    
    # ...gensa, usw.
    
  ))
  
  
  
  
  # benchmark ------------------------------------------------------------------
  , tar_target(rrs_base, {
    set.seed(SEED)
    resample(tasks, baselearners[[1]], resamplings[[1]], store_models = TRUE)
    
  },
  
  pattern = cross(
    slice(map(tasks), 3:4),
    baselearners,
    resamplings
  ), iteration = "list"
  )
  
  
  ,tar_target(rrs, {
    # out = tryCatch(
    #   {
    l0 = tuneablelearners$clone(deep = TRUE)
    l0$fallback = lrn("classif.featureless")
    #print(l0$id)
    learner = tuners[[1]](l0)
    #learner$fallback = lrn("classif.featureless")
    set.seed(SEED) # -> all resamplings have the same train/test split
    resample(
      tasks, learner,
      resamplings[[1]], store_models = TRUE)
    #   },
    #   error = function(cond){
    #     return(NULL)
    #   },
    #   finally = {}
    # )
    # out
  },
  
  pattern = cross(
    slice(map(tasks), 1:9#3:4
    ),
    slice(
      map(tuneablelearners), c(7:8, (1:10)[-(7:8)])
    )
    ,
    slice(tuners, 1:8),
    resamplings),
  iteration = "list")
  
  
  # output ---------------------------------------------------------------------
  ,tar_target(leaderboard, {
    
    leaderboard =
      
      
      #do.call(c, c(rrs_base, rrs))
      as_benchmark_result(rrs)$score() %>%
      select(contains(c("classif", "regr", "_id", "iter"))) %>%
      #select(- resample_result)%>%
      arrange(task_id, classif.ce)
    print(leaderboard)
    leaderboard
  },
  pattern = map(rrs))
  
  ,tar_target(
    archive,
    {out = tryCatch(
      {
        rrs %>%
          extract_inner_tuning_archives() %>%
          arrange(
            task_id,
            classif.ce) %>%
          relocate(
            contains(c(
              "classif", "regr",
              "runtime_learner",
              "task_id",
              "branch"
            ))
          ) %>% select(-resample_result)
      },
      error = function(cond){
        return(NULL)
      },
      finally = {}
    )
    out}, pattern = map(rrs)
  )
)

