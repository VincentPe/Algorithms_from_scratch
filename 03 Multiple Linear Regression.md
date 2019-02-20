
## Multiple linear regression

The formula for MLR is the following:
![image.png](attachment:image.png)

In which X is written in matrix form, being short for:
![image.png](attachment:image.png)

We use the matrix representation because it is easier in terms of writing the formulas and quicker to compute
![image.png](attachment:image.png)

As with the simple linear regression, at local or global minima the slope of the residual plot will equal zero. To determine the values of x that maximize or minimize f(x), take the derivative, set it equal to zero and solve for x.

The sum of squared errors is:
![image.png](attachment:image.png)

For which the errors can actually be replaced by the difference between the predictions and the actuals.
And then the predictions, are just the betas multiplied with the X matrix
![image.png](attachment:image.png)

The function within brackets transposed is just each element inside of it transposed. <br>
After doing that, we can multiply out.
![image.png](attachment:image.png)

Now we need to find the point where the slope is zero for this function, therefore we take the derivative and set equal to 0. <br>
Basic differentiation rules for matrix calculus
![image.png](attachment:image.png)

Since there are only plus and minus signs in the equation (and no divisions or multiplications) we can use the sum rule and perform the derivatives in parts.
![image.png](attachment:image.png)

Now the transpose of the third part (the part in brackets) can be simplified by taking the transpose of the elements within. <br>
We then add the two negative parts, and devide by 2 since both sides have a 2 in them <br>
Taking the X.T* X inverse of both sides will leave just the B.T on the left side. <br>
So if we just transpose both sides one more time, we will have an equation with B on one side <br>
Note that X transposed multiplied with X is symmetric, so transposing it will result in the same
![image.png](attachment:image.png)

Note that this formula can only be performed, if the inverse of (X.T * X) exists. <br>
<br>
Multiplying with an inverse matrix is like division, it is the reciprocal function of multiplication. Thus, what would B.T be if it was not multiplied with (X.T * X). We take the inverse of (X.T * X) to find out. <br>
<br>
If an inverse of this function does not exist, it means there would be multiple solutions. <br>
This in turn means that the columns of X are linear dependent, i.e. one or more columns are a combination of other columns. <br>
<br>
Testing whether an inverse exists, we use Gaussian elimination to transform a matrix by solving for systems of linear equations. The matrix is singular and does not have one (and only one) inverse, if the test fails by leaving zeros that can’t be removed on the leading diagonal. See my IdentifyingSpecialMatrices notebook, for a code example which was an assignment in the Mathematics for Machine Learning Coursera course. <br>
<br>
If an inverse exists we can use the Gauss elimination again, but this time perform operation both on the matrix itself and on an identity matrix (leading diagonal 1's and all others zero) which we will transform to our inverse matrix. <br>
The technique remains straightforward, however, efficiently programming this function remains a challange when the matrix size increases. Find an example of this method below:
![image.png](attachment:image.png)

Note2: If two or multiple columns are not an exact combination of each other but (high levels of) multi-collinearity exists, the overall fit and predictive ability of the model can still work. However, the coefficients might be off as the model cannot calculate how the variance in the dependent variable is correlated with one single independent variable as the single predictor is correlated with another predictor and thus most often moves along. 

## Proof that this is actually true


```python
# %load notebook_preps.py
# Libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error

from IPython.core.interactiveshell import InteractiveShell
import warnings

# Notebook options
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
%pylab inline
pylab.rcParams['figure.figsize'] = (9, 6)

# UDF's
def Variance(x):
    return np.sum((x - np.mean(x))**2) / len(x)

def Covariance(x, y):
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) -1)

def transpose_df(df):
    R, C = df.shape
    df_tr = pd.DataFrame(np.zeros((C, R)))    
    for row in range(C):
        for col in range(R):
            df_tr.iloc[row, col] = df.iloc[col, row]    
    return df_tr

def matrix_dotpr(X, Y):
    R, C = X.shape[0], Y.shape[1]
    df_tr = pd.DataFrame(np.zeros((R, C)))
    for row in range(R):
        for col in range(C):
            df_tr.iloc[row, col] = round(np.sum(X.iloc[row,:] * Y.iloc[:,col]), 6)
    return df_tr
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
from numpy.linalg import inv
from sklearn.datasets import load_boston
from statsmodels.regression.linear_model import OLS
```


```python
# load the boston data set
boston = load_boston()
X = pd.DataFrame(boston.data)
y = boston.target
feature_names = boston.feature_names
```


```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.593761</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.596783</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.647423</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Link to the tutorial: https://datascienceplus.com/linear-regression-from-scratch-in-python/ 
# print(boston.DESCR)
```


```python
# Test UDF's
np.allclose(X.T, transpose_df(X))
np.allclose(matrix_dotpr(X, X.T), X.dot(X.T))
```




    True






    True




```python
# We will need to add a vector of ones (arbitrary, as any constant would suffice) to our feature matrix for the intercept term
# A model most often has a bias and does not originate at 0. Therefore, having one constant feature account for the average bias
# will create a coefficient for the intercept.
int_ = np.ones(shape=y.shape)
X = np.column_stack((int_, X))
```


```python
# calculate coefficients using closed-form solution
coeffs = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
coeffs
```




    array([ 3.64911033e+01, -1.07170557e-01,  4.63952195e-02,  2.08602395e-02,
            2.68856140e+00, -1.77957587e+01,  3.80475246e+00,  7.51061703e-04,
           -1.47575880e+00,  3.05655038e-01, -1.23293463e-02, -9.53463555e-01,
            9.39251272e-03, -5.25466633e-01])




```python
# Check with a known model
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=False)
lm.fit(X, y)
lm.coef_
```




    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)






    array([ 3.64911033e+01, -1.07170557e-01,  4.63952195e-02,  2.08602395e-02,
            2.68856140e+00, -1.77957587e+01,  3.80475246e+00,  7.51061703e-04,
           -1.47575880e+00,  3.05655038e-01, -1.23293463e-02, -9.53463555e-01,
            9.39251272e-03, -5.25466633e-01])




```python

```
