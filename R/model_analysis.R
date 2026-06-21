# SynapseFlow Statistical Analysis — R
# Analyzes benchmark results, generates visualizations
library(jsonlite)

# Load eval data
eval_latest <- fromJSON("../eval/eval_latest.json")
cat("Model Rankings:\n")
for (i in seq_along(eval_latest$rankings)) {
  cat(sprintf("  %d. %s: %.1f%%\n", i, eval_latest$rankings[i],
              eval_latest$results[[eval_latest$rankings[i]]]$overall))
}

# Category comparison
cat("\nCategory Performance Matrix:\n")
models <- names(eval_latest$results)
cats <- names(eval_latest$results[[1]]$categories)
m <- matrix(0, nrow=length(models), ncol=length(cats))
rownames(m) <- models; colnames(m) <- cats
for (model in models) {
  for (cat in cats) {
    m[model, cat] <- eval_latest$results[[model]]$categories[[cat]]$pct
  }
}
print(round(m, 1))

# Best per category
cat("\nBest Model Per Category:\n")
for (cat in cats) {
  best_idx <- which.max(m[, cat])
  cat(sprintf("  %s: %s (%.1f%%)\n", cat, names(best_idx), m[best_idx, cat]))
}
