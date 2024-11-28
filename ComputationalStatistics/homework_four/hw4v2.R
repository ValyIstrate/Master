library(leaps)

read_data_from_file = function(file, has_header) {
  data = read.table(file, has_header)
  
  if (has_header == FALSE) {
    colnames(data) = c(
      "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
      "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"     
    )
  }
  
  data
}

regression = function(data, y) {
  n_var = 13;
  
  formula = as.formula(paste(y, "~ ."))
  subset_fit = regsubsets(formula,  data,
                           nbest = 1, method = "exhaustive", 
                           nvmax = n_var)
  
  summ = summary(subset_fit)
  print(summ)
  
  rss = summ$rss                  
  adj_r2 = summ$adjr2             
  r2 = summ$rsq                   
  cp = summ$cp   
  
  
  metrics = data.frame(
    Num_Predictors = 1:length(rss),
    RSS = rss,
    R2 = r2,
    Adjusted_R2 = adj_r2,
    Cp = cp
  )
  
  metrics$Cp_Difference = abs(metrics$Cp - metrics$Num_Predictors - 1)
  predictors = data[, !(colnames(data) %in% y)]
  variables = apply(summ$which, 1, function(row) {
    colnames(predictors)[row[-1]]  
  })
  
  metrics$Variables = variables
  
  print(metrics)
  
  plot(metrics$Num_Predictors, metrics$RSS, type = "b",
       main = paste("RSS"),
       xlab = "Subset Size", ylab = "Residual Sum of Squares (RSS)",
       pch = 16, col = "red")
  
  plot(metrics$Num_Predictors, metrics$R2, type = "b",
       main = paste("R^2"),
       xlab = "Subset Size", ylab = "R^2",
       pch = 16, col = "red")
  
  plot(metrics$Num_Predictors, metrics$Adjusted_R2, type = "b",
       main = paste("Adjusted R^2"),
       xlab = "Subset Size", ylab = "Adjusted R^2",
       pch = 16, col = "red")
  
  plot(metrics$Num_Predictors, metrics$Cp, type = "b",
       main = paste("Cp"),
       xlab = "Subset Size", ylab = "Mallow's Cp",
       pch = 16, col = "red")
  
  plot(metrics$Num_Predictors, abs(metrics$Cp - metrics$Num_Predictors - 1), type = "b",
       main = paste("Cp Difference"),
       xlab = "Subset Size", ylab = "Mallow's Cp",
       pch = 16, col = "red")
  
  
  best_model_rss = metrics[which.min(metrics$RSS), ]
  best_model_r2 = metrics[which.max(metrics$R2), ] 
  best_model_adj_r2 = metrics[which.max(metrics$Adjusted_R2), ]
  
  
  metrics$Cp_Difference = abs(metrics$Cp - metrics$Num_Predictors - 1)
  
  subset_metrics = metrics[metrics$Cp_Difference == min(metrics$Cp_Difference), ]
  best_model_cp = subset_metrics[which.min(subset_metrics$Cp), ]
  
  metrics_non_zero_cp = metrics[metrics$Cp_Difference != 0, ]
  
  subset_metrics = metrics_non_zero_cp[metrics_non_zero_cp$Cp_Difference == min(metrics_non_zero_cp$Cp_Difference), ]
  best_model_cp = subset_metrics[which.min(subset_metrics$Cp), ]
  
  
  cat("\nBest Model (Min RSS):\n")
  print(best_model_rss)
  
  cat("\nBest Model (Max R^2):\n")
  print(best_model_r2)
  
  cat("\nBest Model (Max Adjusted R^2):\n")
  print(best_model_adj_r2)
  
  cat("\nBest Model (Min Cp difference):\n")
  print(best_model_cp)
  
  
  variables_rss = eval(parse(text = as.character(best_model_rss$Variables)))
  variables_r2 = eval(parse(text = as.character(best_model_r2$Variables)))
  variables_adj_r2 = eval(parse(text = as.character(best_model_adj_r2$Variables)))
  variables_cp = eval(parse(text = as.character(best_model_cp$Variables)))
  
  common_variables = Reduce(intersect, list(variables_rss, variables_r2, variables_adj_r2, variables_cp))
  
  cat("\nBest model:\n")
  print(common_variables)
  
}

ex1 = function() {
  data = read_data_from_file("house.dat", TRUE)
  regression(data, "PRICE")
}

ex2 = function() {
  data = read_data_from_file("boston.dat", FALSE)
  regression(data, "MEDV") 
}
