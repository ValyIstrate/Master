ex1 = function() {
  ex1_array = c(1, 9, 2, 6, 2, 7, 8, 7, 5, 5)
  
  #1.a
  print(sprintf("1.a) -> %f", sum(ex1_array)/10))
  
  #1. b
  for (value in ex1_array) {
    print(sprintf("1.b) -> log_e(%d) = %f", value, log(value, base = exp(1))))
  }
  
  #1. c
  maxx = max(ex1_array)
  minn = min(ex1_array)
  print(sprintf("1.c) -> Min: %d ; Max: %d ; Max - Min = %d", minn, maxx, maxx - minn))
  
  #1. d
  y_array = c()
  for (value in ex1_array) {
    y = (value - 5.2) / 2.740641
    y_array <- append(y_array, y)
    print(sprintf("1.d) -> (%d - 5.2)/2.740641 = %f", value, y))
  }
  
  #1. e
  print(y_array)
  print(sprintf("1.e) -> Mean: %f", mean(y_array)))
  print(sprintf("1.e) -> Standard Deviation: %f", sd(y_array)))
}

ex2 = function() {
  monthly_bills = c(46, 33, 49, 37, 36, 50, 58, 32, 49, 35, 30, 58)
  
  print(sprintf("Total value paid this year: %d", sum(monthly_bills)))
  print(sprintf("Least amount paid in a month: %d", min(monthly_bills)))
  print(sprintf("Biggest amount paid in a month: %d", max(monthly_bills)))
  
  number_of_times_bill_exceeded_40 = 0
  months_bill_exceeded_40 = c()
  for (i in 1:length(monthly_bills)) {
    if (monthly_bills[i] > 40) {
      number_of_times_bill_exceeded_40 = number_of_times_bill_exceeded_40 + 1
      months_bill_exceeded_40 <- append(months_bill_exceeded_40, i)
    }
  }
  print(sprintf("Bills exceeded 40 %d times! This represents %f percent of all bills!",
                number_of_times_bill_exceeded_40,
                (number_of_times_bill_exceeded_40/length(monthly_bills)) * 100))
  cat("Months in which the bill exceeded 40: ", months_bill_exceeded_40)
}

ex3 = function() {
  arr = scan()
  cat("Input array: ", arr)
  
  cat("\nArray minimum: ", min(arr))
  cat("\nArray maximum: ", max(arr))
  cat("\nArray mean: ", mean(arr))
  cat("\nArray median: ", median(arr))
  cat("\nArray standard deviation: ", sd(arr))
  
  arr = sort(arr) 
  cat("\nSorted array: ", arr)
  
  standardized_array = (arr - mean(arr)) / sd(arr)
  cat("\nStandardized array: ", standardized_array)
}