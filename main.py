import pandas as pd
import numpy as np

df = pd.read_excel("assets/athlete_injury_prediction_dataset.xlsx")

# print("Head:", df.head())
# print("Info:", df.info())
# print("Description:", df.describe())

## Missing values handling

df_mdl = df.drop(columns = ["athlete_id", "days_until_injury"])

#check missing values
print("Missing values:\n", df_mdl.isnull().sum())

df_mdl = pd.get_dummies(df_mdl, columns=['gender', 'sport_type'], drop_first=True)

X = df_mdl.drop(columns=['injury_flag'])
y = df_mdl['injury_flag']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)