library(ggplot2)

ex1 = function() {
  file_path = "house.dat"
  
  table_data = read.table(file_path, header = TRUE,
                          sep = "", stringsAsFactors = FALSE)
  
  all_vars = names(table_data)[-1]
  n_vars = length(all_vars)
  
  best_models = list()
  all_criteria = data.frame(Dimension = integer(), RSS = numeric(),
                            R2 = numeric(), Adj_R2 = numeric(), Cp = numeric())
  
  for (p in 1:n_vars) {
    subsets <- combn(all_vars, p, simplify = FALSE)
    metrics_list <- list()
    
    for (subset in subsets) {
      metrics <- compute_metrics(table_data, subset, all_vars)
      metrics_list <- append(metrics_list,
                             list(c(vars = paste(subset, collapse = ","), metrics)))
    }
    
    results_df <- do.call(rbind, lapply(metrics_list, function(x) {
      data.frame(vars = x[1], RSS = as.numeric(x[2]),
                 R2 = as.numeric(x[3]), Adj_R2 = as.numeric(x[4]),
                 Cp = as.numeric(x[5]))
    }))
    
    best_models[[p]] <- list(
      best_rss = results_df[which.min(results_df$RSS), ],
      best_r2 = results_df[which.max(results_df$R2), ],
      best_adj_r2 = results_df[which.max(results_df$Adj_R2), ],
      best_cp = results_df[which.min(results_df$Cp), ]
    )
    
    all_criteria <- rbind(all_criteria,
                          data.frame(Dimension = p,
                                     RSS = min(results_df$RSS),
                                     R2 = max(results_df$R2),
                                     Adj_R2 = max(results_df$Adj_R2),
                                     Cp = min(results_df$Cp)))
  }
  

  generate_plots(all_criteria)
  
  best_overall = do.call(rbind, lapply(best_models, function(x) x$best_adj_r2))
  best_model = best_overall[which.max(best_overall$Adj_R2), ]
  
  print("Best Model WRT Adj_R2:")
  print(best_model)
}

compute_metrics = function(table_data, subset_vars, all_vars) {
  formula = as.formula(paste("PRICE ~", paste(subset_vars, collapse = " + ")))
  model = lm(formula, data = table_data)
  
  rss = sum(residuals(model) ^ 2)
  
  tss = sum((table_data$PRICE - mean(table_data$PRICE)) ^ 2)
  
  r2 = 1 - (rss / tss)
  adj_r2 <- 1 - ((1 - r2) * (nrow(table_data) - 1) / (nrow(table_data) - length(subset_vars) - 1))
  
  sigma2 = sum(residuals(lm(PRICE ~ ., data = table_data)) ^ 2) / (nrow(table_data) - length(all_vars) - 1)
  cp = rss / sigma2 - nrow(table_data) + 2 * length(subset_vars)
  
  return(c(rss = rss, r2 = r2, adj_r2 = adj_r2, cp = cp))
}

generate_plots = function(criteria_data) {
  ggplot(criteria_data, aes(x = Dimension, y = RSS)) +
    geom_line() + geom_point() +
    ggtitle("RSS WRT Model Dimension") +
    xlab("Model Dimensioni") + ylab("RSS")
  
  ggplot(criteria_data, aes(x = Dimension, y = R2)) +
    geom_line() + geom_point() +
    ggtitle("R2 WRT Model Dimension") +
    xlab("Model Dimension") + ylab("R2")
  
  ggplot(criteria_data, aes(x = Dimension, y = Adj_R2)) +
    geom_line() + geom_point() +
    ggtitle("Adj_R2 WT Model Dimension") +
    xlab("Model Dimension") + ylab("Adj_R2")
  
  ggplot(criteria_data, aes(x = Dimension, y = Cp)) +
    geom_line() + geom_point() +
    ggtitle("CP WRT Model Dimension") +
    xlab("Model Dimension") + ylab("CP")
}
