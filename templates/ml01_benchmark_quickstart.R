# Imports ---------------------------------------------------------------------
library(mlr3)
library(mlr3learners)
library(mlr3tuningspaces)
library(mlr3pipelines)
library(dplyr)

# Inputs ----------------------------------------------------------------------
# FIXME: Tasks
tasks_raw = list(tsk("penguins")) # tsk("mtcars") # for regr
# FIXME: Evaluation (outer)
resamplings = list(rsmp("cv", folds = 3))

# Automated Benchmark ----------------------------------------------------------
# - Setup (Optional Inputs: Tuning Parameter, Learners, Hyperparameter) --------
# filter only the first appearing task type
task_type = tasks_raw[[1]]$task_type
tasks_raw = tasks_raw[sapply(tasks_raw, function(task)task$task_type) == task_type]

# - auto preprocessing ---------------------------------------------------------
tasks = lapply(tasks_raw, function(task)ppl("robustify", task = task)$train(task)[[1]])

#' Default Autotuner.
#'
#' @param learner: An (mlr3)-Learner with tune-tokens.
#' @return [pipelrn] autotuner.
#' @export
tuner_def = function(
    learner,
    method      = tnr("random_search"),
    resampling  = rsmp("holdout", ratio = 2/3),
    measure     = NULL, # e.g. msr("classif.acc")
    trm_evals   = 10,
    trm_k       =  1,
    trm_time    = 10 * 60,
    trm_clock   = Sys.time() + 30 * 60,
    #trm_perf    = 0, trm_stag_iters = 1, trm_stag_thresh = 0,
    terminator  = NULL,
    id          = "rs_small",
    store_model = TRUE,
    ...
){
  if(is.null(terminator)){
    terminator = trm("combo", any = TRUE, list(
      trm("evals", n_evals = trm_evals, k = trm_k)
      ,trm("run_time", secs = trm_time)
      ,trm("clock_time", stop_time = trm_clock)
      #,trm("perf_reached", level = trm_perf)
      #,trm("stagnation", iters = trm_stag_iters, threshold = trm_stag_thresh)
    ))
  }
  at = auto_tuner(
    method     = method,
    learner    = learner,
    measure    = measure,
    resampling = resampling,
    terminator = terminator,
    store_model = store_model,
    ...)
  at$id = stringr::str_replace(at$id, "tuned", id)
  at
}

# - Get all default Learners ---------------------------------------------------
if (task_type == "classif"){
  learners = c(
    lrns(paste0("classif.", c("featureless", "naive_bayes"))),
    mlr_tuning_spaces$keys() %>%
      {.[grepl("classif", .) & grepl("default", .)  & !grepl("svm",.)]} %>%
      {lapply(., function(ts) lts(ts)$get_learner())} %>%
      c(lts("classif.svm.default")$get_learner(type = "C-classification")) %>%
      lapply(tuner_def)
  )} else if (task_type == "regr"){
    learners = c(
      lrns(paste0("regr.", c("featureless"))),
      mlr_tuning_spaces$keys() %>%
        {.[grepl("regr", .) & grepl("default", .)  & !grepl("svm",.)]} %>%
        {lapply(., function(ts) lts(ts)$get_learner())} %>%
        c("regr.svm.default" %>% 
            {lts(.)$get_learner(type = to_tune(c("eps-regression", "nu-regression")))}
        ) %>%
        lapply(tuner_def)
    )}
# - Benchmark ------------------------------------------------------------------
# Robustify - learner specific preprocessing, if needed
#bmg = bmg %>% 
#  rowwise() %>% mutate(
#    task = ppl("robustify", task = task, learner = learner)$train(task),
#    learner = list(learner %>% {.$fallback = lrn(paste0(task_type, ".featureless")); .})) %>%
#  data.table::data.table()

# prepare a benchmark grid
bmg = benchmark_grid(tasks, learners, resamplings)

# trigger benchmark
bmr = NULL

if (FALSE) {
  #bmr = benchmark(bmg, store_models = TRUE) # <- one batch or iteratively:
  for (i in (length(bmr) + 1):nrow(bmg)){
    bmr = c(bmr, benchmark(bmg[i,], store_models = TRUE))}
  bmr = do.call(c, bmr)
  
  # save
  qs::qsave(bmr, file = format(Sys.time(), "%Y%m%d-%H%M%S-bmr.qs"))
  # restore
  #bmr = qs::qread(tail(list.files(pattern="bmr.qs"), 1))
  
  # extract results
  bmr$aggregate()
  bmr$score()
  extract_inner_tuning_results(bmr)
  extract_inner_tuning_archives(bmr)
  plot(bmr)
}
