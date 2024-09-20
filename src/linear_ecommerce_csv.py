import math

import pandas as pd
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
from scipy.stats import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# --------------- DataFrame ----------------
pd.options.display.max_columns = None
# pd.options.display.max_rows = None
customers = pd.read_csv('/Users/audrius/Documents/DataScience/ml-regression/data/ecommerce_customers.csv')
print(customers.head())
print(customers.info())
print(customers.describe())
# --------------- Analysis ----------------
sns.pairplot(customers, kind='scatter', plot_kws={'alpha': 0.4}, diag_kws={'alpha': 0.55, 'bins': 40})
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, alpha=0.5)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers, scatter_kws={'alpha': 0.3})
plt.show()
# ---------------- Splitting the Data -------------------
print(customers.info())
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# ----------------- Training The Model using Scikit Learn---------------------
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
lm.score(X, y)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
print(cdf)
# --------------- Training The Model using OLS ----------------
X = sm.add_constant(X_train)
model = sm.OLS(y_train, X)
model_fit = model.fit()
print(model_fit.summary())
# ----------------- Predictions -----------------
predictions = lm.predict(X_test)
print(predictions)
sns.scatterplot(predictions, y_test)
plt.ylabel('Predictions')
plt.title('Yearly Amount Spent vs. Model Predictions')
plt.show()
# -------------------- Evaluation of the model -----------
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, predictions)))
# -------------- Residuals - Normally distributed --------
residuals = y_test - predictions
sns.distplot(residuals, bins=40)
plt.show()
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()
