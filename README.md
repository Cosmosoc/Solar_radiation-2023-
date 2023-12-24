Hello, this is Dev, and here’s my document to explain the python code that I have written for solar predictions.
In my code I have used 3 models- 
1. Random Forest Regressor
2. ExtraTreesRegressor
3. XGBRegressors
Firstly  I preprocessed the data by droping all the rows with null values in them and changing the string data into numerical data, then I calculated the correlation among different features of the data and represented it graphically(using heatmaps and bargraph). 
Then I stored all the data that had correlation with Radiation in X and radiation data in y.
Then I distributed that data among x_train,x_test and y_train,y_test with random state=1 and test size=0.2 cuz It gave the best results.
Then I developed Random Forest Regressor, ExtraTreesRegressor and XGBRegressors models along with calculating there r2 score, mae and rmse which gives a rough idea as to how correctly is the data predicted. 
I also plotted r2 score, mae and rmse via bar graph as to graphically compare the results of the 3 models
I also plotted the 3 most correlated data namely Temperature ,Pressure and Speed against Radiation as to show how correctly are these models predicting the Radiation graphically.
As we can see the predicted data from all 3 models is roughly the same. And also by comparing r2 score Random Forest wins, whereas on comparing mae extra tree regressors win however random forest makes a comeback in rmse value. XGB regressor model is 3rd in every case :< 

