# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

df = pd.read_csv("./solubility_stud.csv", index_col=0)
# %%
Y = df['measured log(solubility:mol/L)']
X1 = df[['MolLogP', 'MolWt']]
X2 = df[['MolLogP', 'NumRotatableBonds']]
X3 = df[['MolLogP', 'AromaticProportion']]

Y.shape
X1.shape
X2.shape
X3.shape

X_train, X_test, y_train, y_test = train_test_split(
    X1, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
print(
    f'y = {model.coef_[0]} x1 + {model.coef_[1]} x2 + {model.intercept_}')
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')

# %%
Y = df['measured log(solubility:mol/L)']
X1 = df[['MolLogP', 'MolWt', 'NumRotatableBonds']]
X2 = df[['MolLogP', 'MolWt', 'AromaticProportion']]

Y.shape
X1.shape
X2.shape

X_train, X_test, y_train, y_test = train_test_split(
    X2, Y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
scaler = Normalizer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
print(
    f'y = {model.coef_[0]} x1 + {model.coef_[1]} x2 + {model.coef_[2]} x3 + {model.intercept_}')
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')
# %%
