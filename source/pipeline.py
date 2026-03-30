import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def run_recommendation(df, target):

    df = df.dropna()

    # Encode categorical
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Detect problem    

    if df[target].dtype == 'object' or df[target].nunique() < 15:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # =========================
    # CLASSIFICATION
    # =========================
    if problem_type == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=8000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = accuracy_score(y_test, preds)

        best_model = max(results, key=results.get)

    # =========================
    # REGRESSION
    # =========================
    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = mean_squared_error(y_test, preds)

        best_model = min(results, key=results.get)

    return problem_type, results, best_model