The main task in the given project was to find the best fit of a machine learning model for solar radiation prediction.

So, I followed the steps below:

1. Data Preprocessing on the CSV file:
   Importing pandas library and reading the solar_prediction CSV file. 
   The missing values in columns were dropped using the dropna() function. These missing values were identified by isnull() and then calculated by sum() function.
2. Data visualisation:
   The correlation heatmap  
3. Correlation:
   The correlation coefficient between temperature and radiation came out to be 0.734. With this correlation coefficient, I proceeded to work on those two variables as one of the features and the target respectively. I have used three models viz Linear Regressor, Random Forest Regressor, and Gradient Boosting Regressor.

4. Training Models: 
   A] Linear Regressor: As this model tries to get a linear relationship, the r2 value for this model did not increase above 0.60. The mse value was higher than expected (above 40000)
   B] Gradient Boosting Regressor: This model captured the relationship between the feature and target better than linear regressor. The r2 value turned out to be 0.734 with decreased mse value, around 25000.
   C] Random Forest Regressor: This model turned out to be the best fit as the r2 value reached 0.93 and mse reduced to 6000 with mae around 32.

