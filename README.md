# Solar Radiation Prediction

## Introduction
Given the following data of the day, this project attempts to predict the amount of solar radiation observed on that particular day<br>
'UNIXTime',<br>
'Date',<br>
'Time',<br>
'Radiation',<br>
'Temperature',<br>
'Pressure', <br>
'Humidity',<br> 
'WindDirection(Degrees)',<br> 
'Speed',<br>
'TimeSunRise',<br>
'TimeSunSet'<br>

## Data preprocessing
Dropping the columns gave a slightly bit more satisfactory result but the code contains <b>IMPUTED</b> data through <b>'mean'</b> strategy incase measurement of one of the features fails someday.

## Feature Selection
The information obtained from 'pearson' method correlation matrix suggests Temperature to be the best feature with a correlation of (0.73).<br>
Other important features are Humidity,Pressure,WindDirection(Degrees),Speed etc
But a new feature can be formed by scaling temperature to its fourth power(i.e temp^4) and this gives a correlation of (0.76).<br>
This is due to domain knowledge of Radiation being proportional to fourth power of temperature as suggested by Stefan-Boltzmann's law.<br><br><br>


I have tried two models: Linear Regression and Random Forest(Regression) and Random forest Regressor turned to be a lot better of the two  <br>
So I have chosen Random Forest as my model and the results for best fit of linear regression are provided as well.

## Linear Regression
Splitting of data was done with 0.3 train/test random=0<br>
xtrain = all the columns except radiation(features describing radiation)<br>
ytrain = column of radiation<br>
Plots may look messy because multiple features have been taken into consideration and we are plotting it in 2D<br>
The model for linear regression was directly imported from sklearn after studying it mathematically and used to fit the data<br>
Results were as follows:<br>
MSE=36074.4039353833<br>
R2 Score=0.6430152713143014<br>
MAE=135.3181548394259<br>
The accuracy was not very good 

## Random Forest Regressor
Splitting of data was done with 0.3 train/test random=0<br>
xtrain = all the columns except radiation(features describing radiation)<br>
ytrain = column of radiation<br>
The model for Random FOrest regressor was directly imported from sklearn.ensemble and used to fit the data<br>
Results were as follows:<br>
MSE=6759.739133035945<br>
R2 Score=0.9331070405289208<br>
MAE=31.882186522633745<br>
The accuracy turned out to be about 93.3% from the R2 score statistic

## Conclusion
This project was my first ML project and given its accuracy it might prove useful in predicting solar flares i.e sudden burst of radiation from sun if the needful data is feeded into the model<br> 
