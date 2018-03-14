library(mlr)

source("measures.R")

# data preprocess ----

targets <- c("l_toxic", "l_severe_toxic", "l_obscene", 
             "l_threat", "l_insult", "l_identity_hate")
colnames(train)[3:8] <- targets

train_all <- X[1:nrow(train),] %>%
  as.matrix() %>%
  as_data_frame() %>%
  cbind(train[, targets]) %>%
  mutate_at(targets, as.logical)

# classifier evaluation ----

toxic_task <- makeMultilabelTask(id = "toxic", data = train_all, target = targets)
toxic_rdesc = makeResampleDesc(method = "CV", stratify = FALSE, iters = 5)

## learners

logreg <- makeLearner("classif.LiblineaRL2LogReg", predict.type = "prob") %>%
  makeMultilabelBinaryRelevanceWrapper()
xg <- makeLearner("classif.xgboost", predict.type = "prob") %>%
  makeMultilabelBinaryRelevanceWrapper()

## resampling

toxic_r = resample(learner = logreg, task = toxic_task, resampling = toxic_rdesc, 
             show.info = TRUE, measures = list(multilabel.f1, macro_roc_auc))
toxic_xg = resample(learner = xg, task = toxic_task, resampling = toxic_rdesc, 
                    show.info = TRUE, measures = list(multilabel.f1, macro_roc_auc))
