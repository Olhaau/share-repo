# uses mlr3 for a simple benchmark.
library("mlr3")

# inputs
task = tsk("iris")
learner = lrn("classif.rpart", keep_model = TRUE)
resampling = rsmp("cv", folds = 3)

# apply learner
rr = resample(task, learner, resampling, store_models = TRUE)

# analyse results
rr$score()
rr$score(predict_set = "train")
rr$aggregate()
mlr3viz::autoplot(rr)

# show resulting models
for(learner in rr$learners) print(learner$model)
mlr3viz::autoplot(rr$learners[[1]])

