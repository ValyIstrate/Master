library(leaps)
if (!require(MASS)) install.packages("MASS")
library(MASS)


house_data <- read.table("house.dat", header = TRUE)

str(house_data)
summary(house_data)

forward_selection = function(data, response) {
  variables = colnames(data)
  variables = variables[variables != response]
  selected_vars = c()
  best_aic = Inf
  
  while (length(variables) > 0) {
    aic_list = c()
    
    for (var in variables) {
      formula = as.formula(paste(response, "~", paste(c(selected_vars, var), collapse = " + ")))
      model = lm(formula, data = data)
      aic_list = c(aic_list, AIC(model))
    }
    
    best_var_index = which.min(aic_list)
    best_var = variables[best_var_index]
    best_aic_new = min(aic_list)
    
    if (best_aic_new < best_aic) {
      selected_vars = c(selected_vars, best_var)
      variables = variables[variables != best_var]
      best_aic = best_aic_new
    } else {
      break
    }
  }
  
  final_formula = as.formula(paste(response, "~", paste(selected_vars, collapse = " + ")))
  final_model = lm(final_formula, data = data)
  return(final_model)
}

backward_selection = function(data, response) {
  variables = colnames(data)
  variables = variables[variables != response]
  best_aic = Inf
  
  while (length(variables) > 1) {
    aic_list = c()
    
    for (var in variables) {
      formula = as.formula(paste(response, "~", paste(variables[variables != var], collapse = " + ")))
      model = lm(formula, data = data)
      aic_list = c(aic_list, AIC(model))
    }
    
    best_var_index = which.min(aic_list)
    best_aic_new = min(aic_list)
    
    if (best_aic_new < best_aic) {
      best_aic = best_aic_new
      variables = variables[-best_var_index]
    } else {
      break
    }
  }
  
  final_formula = as.formula(paste(response, "~", paste(variables, collapse = " + ")))
  final_model = lm(final_formula, data = data)
  return(final_model)
}

stepwise_selection = function(data, response) {
  predictors = setdiff(names(data), response)
  
  current_model = lm(as.formula(paste(response, "~ 1")), data = data)
  best_aic = AIC(current_model)
  selected_vars = c()
  remaining_vars = predictors 
  
  repeat {
    add_candidates = list()
    for (var in remaining_vars) {
      candidate_model = lm(as.formula(paste(response, "~", paste(c(selected_vars, var), collapse = " + "))), data = data)
      add_candidates[[var]] = AIC(candidate_model)
    }
    
    if (length(add_candidates) > 0) {
      best_add = which.min(unlist(add_candidates))
      best_add_var = names(add_candidates)[best_add]
      best_add_aic = add_candidates[[best_add_var]]
      
      if (best_add_aic < best_aic) {
        selected_vars = c(selected_vars, best_add_var)
        remaining_vars = setdiff(remaining_vars, best_add_var)
        best_aic = best_add_aic
        current_model = lm(as.formula(paste(response, "~", paste(selected_vars, collapse = " + "))), data = data)
        next
      }
    }
    
    if (length(selected_vars) > 0) {
      remove_candidates = list()
      for (var in selected_vars) {
        candidate_vars = setdiff(selected_vars, var)
        candidate_model = lm(as.formula(paste(response, "~", paste(candidate_vars, collapse = " + "))), data = data)
        remove_candidates[[var]] <- AIC(candidate_model)
      }
      
      best_remove = which.min(unlist(remove_candidates))
      best_remove_var = names(remove_candidates)[best_remove]
      best_remove_aic = remove_candidates[[best_remove_var]]
      
      if (best_remove_aic < best_aic) {
        selected_vars = setdiff(selected_vars, best_remove_var)
        remaining_vars = c(remaining_vars, best_remove_var)
        best_aic = best_remove_aic
        current_model = lm(as.formula(paste(response, "~", paste(selected_vars, collapse = " + "))), data = data)
        next
      }
    }
    
    break
  }
  
  return(current_model)
}

best_subset_selection = function(data, response) {
  predictors = setdiff(names(data), response)
  
  all_combinations = lapply(1:length(predictors), function(k) {
    combn(predictors, k, simplify = FALSE)
  })
  all_combinations = unlist(all_combinations, recursive = FALSE)
  
  results = data.frame(
    Variables = character(),
    AIC = numeric(),
    R_squared = numeric(),
    Adjusted_R_squared = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (comb in all_combinations) {
    formula_str = paste(response, "~", paste(comb, collapse = " + "))
    model = lm(as.formula(formula_str), data = data)
    
    results = rbind(results, data.frame(
      Variables = paste(comb, collapse = ", "),
      AIC = AIC(model),
      R_squared = summary(model)$r.squared,
      Adjusted_R_squared = summary(model)$adj.r.squared
    ))
  }
  
  best_model_row = results[which.min(results$AIC), ]
  best_model_vars = unlist(strsplit(best_model_row$Variables, ", "))
  best_model = lm(as.formula(paste(response, "~", paste(best_model_vars, collapse = " + "))), data = data)
  
  return (best_model)
}

response <- "PRICE"

forward_model = forward_selection(house_data, response)
backward_model = backward_selection(house_data, response)
stepwise_model = stepwise_selection(house_data, response)
best_subset <- best_subset_selection(data = house_data, response)

full_model = lm(PRICE ~ ., data = house_data)
stepwise_model_second = stepAIC(lm(PRICE ~ 1, data = house_data),
                          scope = list(lower = ~1, upper = full_model),
                          direction = "both")

summary(forward_model)
summary(backward_model)
summary(stepwise_model)
summary(best_subset)

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae = function(actual, predicted) {
  mean(abs(actual - predicted))
}

compare_models <- function(data, response, models) {
  results = data.frame(
    Model = character(),
    AIC = numeric(),
    R2 = numeric(),
    Adjusted_R2 = numeric(),
    RMSE = numeric(),
    MAE = numeric()
  )
  
  for (model_name in names(models)) {
    model = models[[model_name]]
    predicted = predict(model, data)
    actual = data[[response]]
    
    results = rbind(results, data.frame(
      Model = model_name,
      AIC = AIC(model),
      R2 = summary(model)$r.squared,
      Adjusted_R2 = summary(model)$adj.r.squared,
      RMSE = rmse(actual, predicted),
      MAE = mae(actual, predicted)
    ))
  }
  
  return(results)
}

get_model_variables = function(model) {
  terms = attr(terms(model), "term.labels")
  return(terms)
}

forward_vars = get_model_variables(forward_model)
backward_vars = get_model_variables(backward_model)
stepwise_vars = get_model_variables(stepwise_model)
stepwise_vars_second = get_model_variables(stepwise_model_second)
best_subset_vars = get_model_variables(best_subset)

cat("Selected vars (Forward):", paste(forward_vars, collapse = ", "), "\n")
cat("Selected vars (Backward):", paste(backward_vars, collapse = ", "), "\n")
cat("Selected vars (Stepwise):", paste(stepwise_vars, collapse = ", "), "\n")
cat("Selected vars (Best subset):", paste(best_subset_vars, collapse = ", "), "\n")
cat("Selected vars (Stepwise from library):", paste(stepwise_vars_second, collapse = ", "), "\n")

models = list(
  Forward = forward_model,
  Backward = backward_model,
  Stepwise = stepwise_model,
  Best_Subset = best_subset
)

results = compare_models(house_data, "PRICE", models)
print(results)

house_data$y_all = predict(best_subset, house_data)
house_data$y_forward = predict(forward_model, house_data)
house_data$y_backward = predict(backward_model, house_data)
house_data$y_stepwise = predict(stepwise_model, house_data)

y_real = house_data$PRICE

library(ggplot2)

ggplot(house_data, aes(x = y_real)) +
  geom_point(aes(y = y_all, color = "All"), size = 3) +
  geom_point(aes(y = y_forward, color = "Forward"), size = 3) +
  geom_point(aes(y = y_backward, color = "Backward"), size = 3) +
  geom_point(aes(y = y_stepwise, color = "Stepwise"), size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 1) +
  labs(
    title = "Prediction Comparison for the 4 models",
    x = "Real Values (y)",
    y = "Estimated Values (y_hat)",
    color = "Model"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("All" = "red", "Forward" = "blue", 
                                "Backward" = "green", "Stepwise" = "purple"))