from unicodedata import numeric
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv('Solar_Prediction.csv')
print(df.dtypes)
data = df.iloc[:, 4:10]
data.info()
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = pd.DataFrame(imp_mean.fit_transform(data), columns=data.columns)
df.iloc[:, 4:10] = imputed_data
print(df.dtypes)

df.to_csv('Solar_Prediction_Imputed.csv', index=False)
df.info()

numeric_df = df.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

# Plotting heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='rainbow', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#plotting another correlation graph
plt.figure(figsize=(10, 5))
df_corr_solar = correlation_matrix['Radiation'].sort_values(ascending=False)
df_corr_solar.drop('Radiation', inplace=True)
df_corr_solar.plot(kind='bar', color='skyblue')
plt.title('Correlation of Solar Radiation with Other Parameters')
plt.xlabel('Parameters')
plt.ylabel('Correlation')
plt.show()

# Extracting the correlation values for the "Radiation" target variable
radiation_correlation = correlation_matrix.loc['Radiation', :].drop('Radiation')
radiation_correlation_sorted = radiation_correlation.sort_values(ascending=True)

# Plotting the correlation values using a bar graph
plt.figure(figsize=(10, 5))
bars = plt.bar(radiation_correlation_sorted.index, radiation_correlation_sorted, color='green')

# Adding value annotations on top of the bars
for bar, value in zip(bars, radiation_correlation_sorted):
    plt.text(bar.get_x() + bar.get_width() / 1 - 0.15, bar.get_height() + 0.01, f'{value:.2f}', ha='center', color='black')

plt.xlabel('Features')
plt.ylabel('Correlation with Radiation')
plt.title('Correlation between Radiation and Other Features in Ascending Order')
plt.xticks(rotation=30, ha='right')
plt.show()

print("\n\n")


#   X------------------------DESCISION TREE REGRESSOR MODEL------------------------X
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

Y = numeric_df.iloc[:, 4]
X = numeric_df
print(type(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model1 = DecisionTreeRegressor(random_state=4)
model1.fit(X_train, Y_train)
predictions = model1.predict(X_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

print("Making predictions for the following 5 sets:")
print(numeric_df.head())
print("The predictions are")
print(model1.predict(numeric_df))

# Evaluate the model
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test , predictions)

print(f"\nMean Absolute Error: {mae}")
print(f"\nMean Squared Error: {mse}")
print(f"\nR-squared: {r2}")

# Visualize actual vs predicted values using scatter plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=Y_test, y=predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Scatter Plot)')

# Visualize the residuals using a line plot
residuals = Y_test - predictions
plt.subplot(1, 2, 2)
sns.lineplot(x=Y_test, y=residuals, marker='o', linestyle='None', markersize=5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

plt.tight_layout()
plt.show()

#  X------------------------------BAYESIAN LINEAR REGRESSION MODEL----------------------------------X

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

Y = numeric_df.iloc[:, 4]
X = numeric_df.iloc[:, 5]
print(X.shape, Y.shape)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to make them 2D
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Applying polynomial features
degree = 2
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Training the Bayesian Linear Regression model
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train_poly, Y_train)

# Make predictions on the test set
Y_pred, y_pred_std = bayesian_model.predict(X_test_poly, return_std=True)

# Evaluate the model
mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize actual vs predicted values using scatter plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test.flatten(), Y_test, color='blue', label='Actual Values')
plt.scatter(X_test.flatten(), Y_pred, color='red', label='Predicted Values')
plt.fill_between(X_test.flatten(), Y_pred - y_pred_std, Y_pred + y_pred_std, color='orange', alpha=0.2, label='Uncertainty')
plt.xlabel('Temperature')
plt.ylabel('Radiation')
plt.title('Actual vs Predicted Values (Scatter Plot)')
plt.legend()

# Visualize the residuals using a line plot
residuals = Y_test - Y_pred
plt.subplot(1, 2, 2)
plt.plot(X_test.flatten(), residuals, marker='o', linestyle='None', markersize=5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Temperature')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

plt.tight_layout()
plt.show()
