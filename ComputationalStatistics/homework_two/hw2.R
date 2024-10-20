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
    
    plot(x, y, type = "h", col = "blue", lwd = 2, 
         main = paste("Distributia binomiala B(", n, ",", p, ")"),
         xlab = "Valori ale lui x", ylab = "Probabilitate")
    
    dev.off()
  }
}

ex3 = function(mu, sigma_values) {
  x = seq(-7, 7, length.out = 1000)
  
  colors = c("red", "green", "blue")
  
  plot(x, dnorm(x, mean = mu, sd = sigma_values[1]), type = "l", col = colors[1], 
       lwd = 2, ylim = c(0, 1), 
       main = expression(paste("Densitatea normala N(", mu, ", ", sigma^2, ") pentru diferite valori ale ", sigma)),
       xlab = "x", ylab = "Densitate")
  
  for (i in 2:length(sigma_values)) {
    lines(x, dnorm(x, mean = mu, sd = sigma_values[i]), col = colors[i], lwd = 2)
  }
  
  legend("topright", legend = paste("σ =", sigma_values), col = colors, lwd = 2)
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
  
  hist(CLT(1), main = "Histograma mediilor, n = 1", xlab = "Medii", col = "lightblue", breaks = 20)
  
  hist(CLT(5), main = "Histograma mediilor, n = 5", xlab = "Medii", col = "lightgreen", breaks = 20)
  
  hist(CLT(10), main = "Histograma mediilor, n = 10", xlab = "Medii", col = "lightcoral", breaks = 20)
  
  hist(CLT(100), main = "Histograma mediilor, n = 100", xlab = "Medii", col = "lightgoldenrod", breaks = 20)
}

### Exercitiul 4 c)
CLT_binom = function(n, size, prob) {
  sample_means = numeric(1000)
  
  for (i in 1:1000) {
    sample = rbinom(n, size = size, prob = prob)
    sample_means[i] = mean(sample)
  }
  
  return(sample_means)
}

ex4_2 = function() {
  par(mfrow = c(2, 2))
  
  hist(CLT_binom(1, 20, 0.5), main = "Histograma mediilor, n = 1", xlab = "Medii", col = "lightblue", breaks = 20)
  
  hist(CLT_binom(5, 20, 0.5), main = "Histograma mediilor, n = 5", xlab = "Medii", col = "lightgreen", breaks = 20)
  
  hist(CLT_binom(10, 20, 0.5), main = "Histograma mediilor, n = 10", xlab = "Medii", col = "lightcoral", breaks = 20)
  
  hist(CLT_binom(100, 20, 0.5), main = "Histograma mediilor, n = 100", xlab = "Medii", col = "lightgoldenrod", breaks = 20)
}