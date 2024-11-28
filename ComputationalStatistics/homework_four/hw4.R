ex1_plots = function(results_df) {
  results_df$Subset_Size = sapply(strsplit(results_df$Variables, ","), length)
  avg_metrics = aggregate(. ~ Subset_Size, data = results_df[, c("Subset_Size", "RSS", "R2", "Adj_R2", "Cp")], mean)
  
  par(mfrow = c(2, 2))
  
  plot(avg_metrics$Subset_Size, avg_metrics$RSS, type = "b", pch = 19, col = "red",
       main = "RSS by Subset Size", xlab = "Subset Size", ylab = "RSS", lwd = 2)
  
  plot(avg_metrics$Subset_Size, avg_metrics$R2, type = "b", pch = 19, col = "red",
       main = "R2 by Subset Size", xlab = "Subset Size", ylab = "R2", lwd = 2)
  
  plot(avg_metrics$Subset_Size, avg_metrics$Adj_R2, type = "b", pch = 19, col = "red",
       main = "Adj R2 by Subset Size", xlab = "Subset Size", ylab = "Adj R2", lwd = 2)
  
  plot(avg_metrics$Subset_Size, avg_metrics$Cp, type = "b", pch = 19, col = "red",
       main = "Cp by Subset Size", xlab = "Subset Size", ylab = "Cp", lwd = 2)
}

compute_metrics = function(table_data, subset_vars) {
  formula = as.formula(paste("PRICE ~", paste(subset_vars, collapse = " + ")))
  model = lm(formula, data = table_data)
  
  rss = sum(residuals(model)^2)
  tss = sum((table_data$PRICE - mean(table_data$PRICE))^2)
  r2 = 1 - (rss / tss)
  adj_r2 = 1 - ((1 - r2) * (nrow(table_data) - 1) / (nrow(table_data) - length(subset_vars) - 1))
  
  full_model = lm(PRICE ~ ., data = table_data)
  sigma2 = sum(residuals(full_model)^2) / (nrow(table_data) - ncol(table_data))
  cp = rss / sigma2 - nrow(table_data) + 2 * length(subset_vars)
  
  return(c(rss = rss, r2 = r2, adj_r2 = adj_r2, cp = cp))
}

ex1_manual = function() {
  file_path = "house.dat"
  table_data = read.table(file_path, header = TRUE, sep = "", stringsAsFactors = FALSE)
  all_vars = names(table_data)[-1]  
  n_vars = length(all_vars)
  
  results = list()
  
  for (p in 1:n_vars) {
    subsets = combn(all_vars, p, simplify = FALSE)  
    for (subset in subsets) {
      metrics = compute_metrics(table_data, subset)
      results = append(results, list(
        list(
          vars = paste(subset, collapse = ","),
          subset_size = p,
          rss = metrics["rss"],
          r2 = metrics["r2"],
          adj_r2 = metrics["adj_r2"],
          cp = metrics["cp"]
        )
      ))
    }
  }
  
  results_df = do.call(rbind, lapply(results, function(x) {
    data.frame(
      Variables = x$vars,
      Subset_Size = as.numeric(x$subset_size),
      RSS = as.numeric(x$rss),
      R2 = as.numeric(x$r2),
      Adj_R2 = as.numeric(x$adj_r2),
      Cp = as.numeric(x$cp)
    )
  }))
  
  best_model = results_df[which.max(results_df$Adj_R2), ]
  
  print("Best model:")
  print(best_model)
  
  results = (list(results_df = results_df, best_model = best_model))
  ex1_plots(results$results_df)
  results
}
