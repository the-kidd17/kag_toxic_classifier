library(reticulate)

use_python("/Users/patrickding/anaconda3/bin/python")
sk <- reticulate::import("sklearn")
np <- reticulate::import("numpy")

roc_auc_fun <- function(task, model, pred, feats, extra.args) {
  # scikit-learn f1 score
  
  sk$metrics$roc_auc_score(y_true = getPredictionTruth(pred), 
                           y_score = as.matrix(getPredictionProbabilities(pred)), 
                           average = extra.args$average)
}

micro_roc_auc <- makeMeasure(id = "micro.roc.auc", name = "micro roc auc score", 
                             minimize = FALSE, 
                             properties = c("classif", "classif.multi", "multilabel"),
                             best = 1, worst = 0, fun = roc_auc_fun,
                             extra.args = list(average = "micro"))
macro_roc_auc <- makeMeasure(id = "macro.roc.auc", name = "macro roc auc score", 
                             minimize = FALSE, 
                             properties = c("classif", "classif.multi", "multilabel"),
                             best = 1, worst = 0, fun = roc_auc_fun,
                             extra.args = list(average = "macro"))