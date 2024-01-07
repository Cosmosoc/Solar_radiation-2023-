# Solar Radiation Prediction

## Aim
To predict the level of solar radiation based on the dataset given 

## Problem Statement
The dataset given contains parameters such as UNIX Time,Date,Time,Radiation,Pressure,Temperature,Humidity,Wind Direction,Speed,TimeSunRise and TimeSunSet.It contains measurements for 4 months in the year 2016 and we had to predict the level of solar radiation for the same.

## Steps undertaken in this project

1. Data Preprocessing
   a. Firstly, I preprocessed the given raw data into meaningful numeric data for increasing the ease as well as efficiency of correlating the given parameters with the radiation.
   b. Secondly, I also got rid of those observations where one or more parameters had null values. This helped in preventing any kind of errors which would have occured during correlation and hence we wouldn't have received the desired result i.e. getting the level of solar radiation with the highest prediction.

2. Correlation
   In this part I found out how data(in general) and radiation(in particular) is correlated with different parameters with the help of a heatmap. It was seen that radiation had the most correlation with temperature and the least with the parameters hour and minute. No correlation was observed with the parameters year and sunsetHour  Hence, I have considered the effect of all parameters except year and sunsetHour while using the models.

3. Use of ML models
   To check the performance of my dataset I used some ML models and calculated their R2 SCORE,MAE and MSE values and conducted my comparison on the basis of these values.
   The models I used were:
   a. KNN Regressor
   b. Decision Tree Regressor
   c. Random Forest Regressor

   For the above models, to create the plot, I have considered the values as such
   a_train: all parameters except radiation,year,sunsetHour
   b_train: radiation

## Tools used
Python,Numpy,Pandas,Seaborn,Sklearn,Matplotlib

## Conclusion
  I chose the Random Forest Regressor model for this problem as it gave me the highest accuracy(R2 Score) of 94.5% with MAE=27.91268679125752 and MSE=5427.355139721152.