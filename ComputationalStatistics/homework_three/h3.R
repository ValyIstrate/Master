ex1 = function() {
  file_path = "alcool.dat"
  table_data = read.table(file_path, header = TRUE,
                          sep = "", stringsAsFactors = FALSE)
  head(table_data)
  
  plot(table_data$Alcool_din_vin, table_data$Decese_datorate_afectiunilor_cardiace,
       xlab = "% of alcohol in Wine (liters/person)",
       ylab = "Deaths caused by heart disease(per 100K people)",
       main = "Scatter Plot",
       pch = 16, col = "blue")
  reg_line = lm(table_data$Decese_datorate_afectiunilor_cardiace 
                ~ table_data$Alcool_din_vin)
  abline(reg_line, col = "red", lwd = 2)
  
  corelation = cor(table_data$Alcool_din_vin,
                   table_data$Decese_datorate_afectiunilor_cardiace)
  
  # If corelation is closer to -1, it means there is a strong negative
  # corelation between the 2 variables (when one goes high, the other goes low)
  
  # If corelation is closer to 1, it means that when one variable goes high,
  # the other does, too
  
  # If corelation is closer to 0, we have no significat corelation between
  # the 2 variables
  corelation
}

ex2 = function() {
  file_path = "iq.dat"
  table_data = read.table(file_path, header = TRUE,
                          sep = "", stringsAsFactors = FALSE)
  head(table_data)
  
  plot(table_data$IQ, table_data$Nota,
       xlab = "Student IQ",
       ylab = "Student Grade",
       main = "Scatter Plot",
       pch = 16, col = "blue")
  reg_line = lm(table_data$Nota ~ table_data$IQ)
  abline(reg_line, col = "red", lwd = 2)
  
  # From the course: Yi = α + βXi + εi
  # Found 2 ways to do this
  summary(reg_line)
  intercept = coef(reg_line)[1]
  coef_IQ = coef(reg_line)[2]
  
  n115 = intercept + coef_IQ * 115
  n130 = intercept + coef_IQ * 130
  
  cat("The predicted grade for a student with IQ=115 is ->", n115, "\n")
  cat("The predicted grade for a student with IQ=130 is ->", n130, "\n")
}

fun_ex3 = function(m, a, b, xmin, xmax, sigma) {
  x = runif(m, min = xmin, max = xmax)
  epsilon = rnorm(m, mean = 0, sd = sigma)
  y = a + b * x + epsilon
  data.frame(x = x, y = y)
}

use_fun_ex3 = function() {
  set.seed(123)
  obs = fun_ex3(100, 2, 5, 0, 10, 1)
  
  print(head(obs))
  
  plot(obs$x, obs$y,
       xlab = "X",
       ylab = "Y",
       main = "Scatter Plot",
       pch = 16, col = "blue")
  
  reg_line = lm(y ~ x, data = obs)
  abline(reg_line, col = "red", lwd = 2)
}

fun_ex4 = function(generated_data) {
  model = lm(y ~ x, data = generated_data)
  coefs = summary(model)$coefficients
  
  print(summary(model))
  
  a_hat = coefs[1, 1]  
  b_hat = coefs[2, 1]  
  
  intervals = confint(model, level = 0.95)
  
  return(list(
    a_hat = a_hat,
    b_hat = b_hat,
    intervals_a = intervals[1, ],  
    intervals_b = intervals[2, ]
  ))
}

use_fun_ex4 = function() {
  obs = fun_ex3(100, 2, 5, 0, 10, 1)
  fun_ex4(obs)
}

print_results = function(case_idx, generated_data) {
  cat("Case", idx, ":\n")
  cat("Coef a (INTERCEPT) est.:", round(results$a_hat, 3), "\n")
  cat("Coef b est.:", round(results$b_hat, 3), "\n")
  cat("95% Confidence Interval for a:", round(results$intervals_a[1], 3),
      "-", round(results$intervals_a[2], 3), "\n")
  cat("95% Confidence Interval for b:", round(results$intervals_b[1], 3),
      "-", round(results$intervals_b[2], 3), "\n\n")
}

generate_and_save_pdf = function(case_idx, a, b, generated_data) {
  pdf_file_name = paste0("output_graph_case_", idx, ".pdf")
  pdf(pdf_file_name)
  
  plot(generated_data$x, generated_data$y,
       xlab = "X",
       ylab = "Y",
       main = paste("Scatter Plot for Case", idx),
       pch = 16, col = "blue")
  
  reg_line = lm(y ~ x, data = generated_data)
  abline(reg_line, col = "red", lwd = 2)
  abline(a, b, col = "green", lwd = 2, lty = 2)
  
  legend("topright", legend = c("Obs", "Estimation", "Real Model"), 
         col = c("blue", "red", "green"), pch = c(16, NA, NA), lty = c(NA, 1, 2))
  
  dev.off()
}

ex5 = function() {
  m_values = c(100, 10, 10000, 10, 10000, 10)
  xmin_values = c(-200, -5, -5, 5, 5, 5)
  xmax_values = c(200, 5, 5, 5.2, 5.2, 5.2)
  sigma_values = c(1.5, 1, 1, 1, 1, 0.01)
  a = 10
  b = 0.8
  
  for (idx in 1:length(m_values)) {
    generated_data = fun_ex3(m_values[idx], a, b,
                             xmin_values[idx], 
                             xmax_values[idx],
                             sigma_values[idx])
    results = fun_ex4(generated_data)
    print_results(idx, results)
  }
}
