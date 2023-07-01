import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import pickle


RANDOM_STATE = 42

DATASET_PATH = "https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/cars.csv"

# загрузка данных
df = pd.read_csv(DATASET_PATH)

mean = df['seats'].mean()
df['seats'].fillna(int(mean), inplace=True)

preprocess_func = lambda x: float(x.split()[0]) if isinstance(x, str) and x[0].isdigit() else None

for column in ['mileage', 'engine', 'max_power']:
    df[column] = df[column].apply(preprocess_func)
    mean = df[column].notna().mean()
    df[column].fillna(mean, inplace=True)



df['torque'].fillna("unknown", inplace=True)


X = df.drop(['selling_price'], axis=1) #матрица объект-признак

y = df['selling_price'] # целевая переменная (target)

X.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner', 'torque'], axis=1, inplace=True)

X = X.join(pd.get_dummies(df['fuel']))
X = X.join(pd.get_dummies(df['seller_type']))
X = X.join(pd.get_dummies(df['transmission']))
X = X.join(pd.get_dummies(df['owner']))


ss = MinMaxScaler()
ss.fit(X) # вычислить min, max по каждому столбцу

X = pd.DataFrame(ss.transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

y_train = np.log(y_train)
y_test = np.log(y_test)

model = Ridge(alpha=0.1)

model.fit(X_train, y_train)

pred = model.predict(X_test)

with open('model.pickle', 'wb') as f:
  pickle.dump(model, f)