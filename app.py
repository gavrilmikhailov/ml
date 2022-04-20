import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from xgboost import XGBClassifier, DMatrix

def to_numeric(str):
    return pd.to_numeric(str[2:])

train = pd.read_csv("kaggle/input/flight/flight_delays_train.csv")
test = pd.read_csv("kaggle/input/flight/flight_delays_test.csv")

train["dep_delayed_15min"] = train["dep_delayed_15min"].map({ "Y": 1, "N": 0 }).values

train["Route"] = train["Origin"] + "_" + train["Dest"]
test["Route"] = test["Origin"] + "_" + test["Dest"]

train.drop(["Dest"], axis=1, inplace=True)
test.drop(["Dest"], axis=1, inplace=True)

train.drop(["Origin"], axis=1, inplace=True)
test.drop(["Origin"], axis=1, inplace=True)

train["Month"] = train["Month"].map(to_numeric).values
test["Month"] = test["Month"].map(to_numeric).values

train["DayofMonth"] = train["DayofMonth"].map(to_numeric).values
test["DayofMonth"] = test["DayofMonth"].map(to_numeric).values

train["DayOfWeek"] = train["DayOfWeek"].map(to_numeric).values
test["DayOfWeek"] = test["DayOfWeek"].map(to_numeric).values

print(train.head())
print(test.head())

X = train.drop("dep_delayed_15min", axis=1).copy()
y = train["dep_delayed_15min"].copy()

X_encoded = pd.get_dummies(X, columns=[
    "UniqueCarrier",
    "Route"
])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)
print("Delayed percentage:")
print("whole dataset", sum(y) / len(y))
print("train part", sum(y_train) / len(y_train))
print("test part", sum(y_test) / len(y_test))

clf_xgb = XGBClassifier(objective="binary:logistic", missing=None, seed=42, early_stopping_rounds=10, eval_metric="aucpr")
clf_xgb.fit(
    X_train, 
    y_train,
    verbose=True,
    eval_set=[(X_test, y_test)]
)

param_grid = {
    "max_depth": [4],
    "learning_rate": [0.1],
    "gamma": [0,14, 0,15, 0,16],
    "reg_lambda": [1.0],
    "scale_pos_weight": [1]
}

# optimal_params = GridSearchCV(
#     estimator=XGBClassifier(
#         objective="binary:logistic",
#         seed=42,
#         subsample=0.2,
#         colsample_bytree=0.5,
#         early_stopping_rounds=5,
#         eval_metric="auc"
#     ),
#     param_grid=param_grid,
#     scoring="roc_auc",
#     verbose=0,
#     n_jobs=-1,
#     cv=3
# )
# optimal_params.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
# print(optimal_params.best_params_)

clf_xgb = XGBClassifier(
    objective="binary:logistic", 
    gamma=0.14,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=1.0,
    scale_pos_weight = 1,
    seed=42,
    colsample_bytree=0.5,
    early_stopping_rounds=5, 
    eval_metric="aucpr"
)

clf_xgb.fit(
    X_train,
    y_train,
    verbose=True,
    eval_set=[(
        X_test,
        y_test
    )]
)

# test_encoded = pd.get_dummies(test, columns=[
#     "UniqueCarrier",
#     "Route"
# ])

result = clf_xgb.predict(test)
# result.to_csv("submission.csv")