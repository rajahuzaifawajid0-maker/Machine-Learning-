import pandas as pd



df = pd.read_csv("Titanic-Dataset.csv")
print(df)


print(df.isnull().sum())


df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Cabin"] = df["Cabin"].fillna("Unknown")

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

df["Embarked"] = le.fit_transform(df["Embarked"])

print(df.dtypes)

df["Age"] = df["Age"].astype(int)

df["Fare"] = df["Fare"].astype(int)

print(df.dtypes)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))

df[["Age","Fare","PassengerId","Survived","Pclass","Sex","Embarked"]] = scaler.fit_transform(
    df[["Age","Fare","PassengerId","Survived","Pclass","Sex","Embarked"]]
)

print(df)

x = df[["Age","Fare","PassengerId","Pclass","Sex","Embarked"]]
y = df["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

df.to_csv("cleaned_titanic.csv", index=False)

import os
print(os.path.abspath("cleaned_titanic.csv"))