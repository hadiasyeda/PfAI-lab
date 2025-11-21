import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor




def rmse_cv(model, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rmse = -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
    return np.sqrt(rmse)


def main():

    
    if not (os.path.exists("train.csv") and os.path.exists("test.csv")):
        print("ERROR: train.csv or test.csv missing!")
        return

    
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    
    y = np.log1p(train["SalePrice"])   # log transform
    train = train.drop(columns=["SalePrice"])

    
    full = pd.concat([train, test], axis=0)

    
    numeric_cols = full.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = full.select_dtypes(include=["object"]).columns

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    
    full_processed = preprocessor.fit_transform(full)

    X_train = full_processed[:len(train)]
    X_test = full_processed[len(train):]

    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    lasso = LassoCV(cv=5)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    print("\nEvaluating Models:\n")

    ridge_rmse = rmse_cv(ridge, X_train, y)
    print("Ridge RMSE:", ridge_rmse.mean())

    lasso_rmse = rmse_cv(lasso, X_train, y)
    print("Lasso RMSE:", lasso_rmse.mean())

    rf_rmse = rmse_cv(rf, X_train, y)
    print("Random Forest RMSE:", rf_rmse.mean())

    
    ridge.fit(X_train, y)
    lasso.fit(X_train, y)
    rf.fit(X_train, y)

    
    preds = (
        0.4 * ridge.predict(X_test) +
        0.4 * lasso.predict(X_test) +
        0.2 * rf.predict(X_test)
    )

    
    final_preds = np.expm1(preds)

    
    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": final_preds
    })

    submission.to_csv("submission.csv", index=False)

    print("\nSubmission file generated successfully as: submission.csv")



if __name__ == "__main__":
    main()