ex1 = function(a, b) {
  x = seq(a, b, length.out = 1000)
  y = sin(x)
  
  plot(x, y, type = "l", col = "red", lwd = 2, 
       main = paste("f(x)= sin(x), x in [", a, ",", b, "]"), 
       xlab = "x", ylab = "f(x)")
}

ex2 = function(n) {
  p_values = seq(0.1, 0.9, by = 0.1)
  for (p in p_values) {
    file_name = paste0("binomial_n", n, "_p", p, ".png")
    
    png(file_name)
    x = 0:n
    
    y = dbinom(x, size = n, prob = p)
    
    plot(x, y, type = "h", col = "red", lwd = 2, 
         main = paste("D-Binom B(", n, ",", p, ")"),
         xlab = "Values of X", ylab = "Prob")
    
    dev.off()
  }
}

# Run with: ex3(0, c(0.5, 1, 2))
ex3 = function(mu, sigma_values) {
  x = seq(-7, 7, length.out = 1000)
  
  colors = c("red", "green", "blue")
  
  plot(x, dnorm(x, mean = mu, sd = sigma_values[1]), type = "l", col = colors[1], 
       lwd = 2, ylim = c(0, 1), 
       main = expression(paste("Normal Density N(", mu, ", ", sigma^2, ") for diferent values of ", sigma)),
       xlab = "x", ylab = "Density")
  
  for (i in 2:length(sigma_values)) {
    lines(x, dnorm(x, mean = mu, sd = sigma_values[i]), col = colors[i], lwd = 2)
  }
  
  legend("topright", legend = paste("sigma =", sigma_values), col = colors, lwd = 2)
}

### Exercitiul 4 a)
CLT = function(n) {
  sample_means = numeric(1000)
  
  for (i in 1:1000) {
    sample = runif(n, min = 0, max = 20)
    sample_means[i] = mean(sample)
  }
  
  return(sample_means)
}

### Exercitiul 4 b)
ex4_1 = function() {
  par(mfrow = c(2, 2))
  
  hist(CLT(1), main = "Mean Histogram, n = 1", xlab = "Means", col = "lightblue", breaks = 20)
  hist(CLT(5), main = "Mean Histogram, n = 5", xlab = "Means", col = "lightgreen", breaks = 20)
  hist(CLT(10), main = "Mean Histogram, n = 10", xlab = "Means", col = "lightcoral", breaks = 20)
  hist(CLT(100), main = "Mean Histogram, n = 100", xlab = "Means", col = "lightgoldenrod", breaks = 20)
}

### Exercitiul 4 c)
CLT_binom = function(n) {
  sample_means = numeric(1000)
  
  for (i in 1:1000) {
    sample = rbinom(n, size = 20, prob = 0.9)
    sample_means[i] = mean(sample)
  }
  
  return(sample_means)
}

ex4_2 = function() {
  par(mfrow = c(2, 2))
  
  hist(CLT_binom(1), main = "Mean Histogram, n = 1", xlab = "Means", col = "lightblue", breaks = 20)
  hist(CLT_binom(5), main = "Mean Histogram, n = 5", xlab = "Means", col = "lightgreen", breaks = 20)
  hist(CLT_binom(10), main = "Mean Histogram, n = 10", xlab = "Means", col = "lightcoral", breaks = 20)
  hist(CLT_binom(100), main = "Mean Histogram, n = 100", xlab = "Means", col = "lightgoldenrod", breaks = 20)
}