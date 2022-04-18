# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./solubility_stud.csv", index_col=0)
Y = df['measured log(solubility:mol/L)']
X1 = df[['MolLogP']]
X2 = df[['MolWt']]
X3 = df[['NumRotatableBonds']]
X4 = df[['AromaticProportion']]

Y.shape
X1.shape
X2.shape
X3.shape
X4.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.xlabel('MolLogP')
plt.ylabel('measured log(solubility:mol/L)')
plt.show()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'y = {model.coef_[0]}x + {model.intercept_}')

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

df_y_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_y_comp

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, y_train_pred, 'r')
plt.show()


print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')

# %%
X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.2)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.xlabel('MolLogP')
plt.ylabel('measured log(solubility:mol/L)')
plt.show()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'y = {model.coef_[0]}x + {model.intercept_}')

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

df_y_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_y_comp

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, y_train_pred, 'r')
plt.show()


print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')

# %%
X_train, X_test, y_train, y_test = train_test_split(X3, Y, test_size=0.2)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.xlabel('MolLogP')
plt.ylabel('measured log(solubility:mol/L)')
plt.show()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'y = {model.coef_[0]}x + {model.intercept_}')

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

df_y_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_y_comp

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, y_train_pred, 'r')
plt.show()


print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')

# %%
X_train, X_test, y_train, y_test = train_test_split(X4, Y, test_size=0.2)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.xlabel('MolLogP')
plt.ylabel('measured log(solubility:mol/L)')
plt.show()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'y = {model.coef_[0]}x + {model.intercept_}')

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

df_y_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_y_comp

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, y_train_pred, 'r')
plt.show()


print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(
    f'Root Mean Squared Error: { np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2(train): {r2_score(y_train, y_train_pred)}')
print(f'R2(test): {r2_score(y_test, y_pred)}')
# print(X4)
# %%
