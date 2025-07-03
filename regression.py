from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import load_boston, eval_metrics

def basic_models(X_train1, X_test1, y_train1, y_test1):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR()
    }
    for name, model in models.items():
        model.fit(X_train1, y_train1)
        preds1 = model.predict(X_test1)
        mse, r2 = eval_metrics(y_test1, preds1)
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

def tuned_models(X_train2, X_test2, y_train2, y_test2):
    """Grid-search, then train and evaluate regressors."""
    tuned = {
        "Ridge": (Ridge(), {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky']
        }),
        "Random Forest": (RandomForestRegressor(), {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        "SVR": (SVR(), {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        })
    }
    for name, (model, params) in tuned.items():
        grid = GridSearchCV(model, params, cv=3, scoring='r2')
        grid.fit(X_train2, y_train2)
        best = grid.best_estimator_
        preds = best.predict(X_test2)
        mse, r2 = eval_metrics(y_test2, preds)
        print(f"{name} (tuned) — MSE: {mse:.2f}, R²: {r2:.2f}, params: {grid.best_params_}")

if __name__ == "__main__":
    df = load_boston()
    X, y = df.drop("MEDV", axis=1), df["MEDV"]
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("----- Model Comparision are: -----")
    basic_models(X_tr1, X_te1, y_tr1, y_te1)
    	
    print("\n----- Tuned Models are:-----")
    tuned_models(X_tr1, X_te1, y_tr1, y_te1)