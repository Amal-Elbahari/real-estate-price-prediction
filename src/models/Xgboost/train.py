import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sqlalchemy import create_engine
import joblib
from scipy import sparse
import matplotlib.pyplot as plt

# ---------------- 1. Load Data ----------------
engine = create_engine("postgresql+psycopg2://postgres:2024@localhost:5432/real")
df = pd.read_sql_query("SELECT * FROM combine", engine)

# ---------------- 2. Data Cleaning ----------------
df.dropna(subset=["price", "surface"], inplace=True)
df = df[
    (df['price'].between(df['price'].quantile(0.01), df['price'].quantile(0.99))) &
    (df['surface'].between(df['surface'].quantile(0.01), df['surface'].quantile(0.99)))
]
df["price_log"] = np.log1p(df["price"])

# ---------------- 3. Feature Engineering ----------------
df["rooms_per_surface"] = df["rooms"] / df["surface"].replace(0, 1)
df["bathrooms_per_surface"] = df["bathrooms"] / df["surface"].replace(0, 1)
df["bedrooms_per_surface"] = df["bedrooms"] / df["surface"].replace(0, 1)

df["surface_x_rooms"] = df["surface"] * df["rooms"]
df["surface_x_bathrooms"] = df["surface"] * df["bathrooms"]
df["surface_x_bedrooms"] = df["surface"] * df["bedrooms"]

df["location_city"] = df["location"].str.split(",").str[0].str.strip()
df["location_district"] = df["location"].str.split(",").str[1].str.strip().fillna("Other")

df["luxury_features_count"] = df[["concierge", "pool", "security", "garden"]].fillna(0).sum(axis=1)

# ---------------- 4. Define Features ----------------
target_encoded_features = ["location_city", "location_district"]
numeric_features = [
    "surface", "rooms", "bedrooms", "bathrooms",
    "rooms_per_surface", "bathrooms_per_surface", "bedrooms_per_surface",
    "surface_x_rooms", "surface_x_bathrooms", "surface_x_bedrooms",
    "luxury_features_count"
]
binary_features = ["terrace", "garage", "elevator", "concierge", "pool", "security", "garden"]
categorical_features_oh = ["property_category"]

# ---------------- 5. Metrics ----------------
def accuracy_10pct(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    return np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp < 0.1)

def evaluate_model(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    return {
        'accuracy_10%': np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp < 0.1),
        'accuracy_15%': np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp < 0.15),
        'accuracy_20%': np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp < 0.2),
        'mape': np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }

# ---------------- 6. Training Function ----------------
def train_model_for_type(df_sub, type_name):
    print(f"\n🚀 Training model for: {type_name} ({len(df_sub)} samples)")

    X = df_sub[target_encoded_features + numeric_features + binary_features + categorical_features_oh]
    y = df_sub["price_log"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Optuna objective ----
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "gamma": trial.suggest_float("gamma", 0, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
            "seed": 42,
            "verbosity": 0,
        }
        num_boost_round = trial.suggest_int("n_estimators", 500, 1200)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold for stability
        scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            te = TargetEncoder(smoothing=0.3, min_samples_leaf=20)
            X_train_te = te.fit_transform(X_train_fold[target_encoded_features], y_train_fold)
            X_val_te = te.transform(X_val_fold[target_encoded_features])

            ct = ColumnTransformer([
                ("num", StandardScaler(), numeric_features),
                ("bin", "passthrough", binary_features),
                ("cat_oh", OneHotEncoder(handle_unknown="ignore"), categorical_features_oh),
            ])

            X_train_rest = ct.fit_transform(X_train_fold)
            X_val_rest = ct.transform(X_val_fold)

            if sparse.issparse(X_train_rest):
                X_train_t = sparse.hstack([X_train_rest, X_train_te.values])
                X_val_t = sparse.hstack([X_val_rest, X_val_te.values])
            else:
                X_train_t = np.hstack([X_train_rest, X_train_te.values])
                X_val_t = np.hstack([X_val_rest, X_val_te.values])

            dtrain = xgb.DMatrix(X_train_t, label=y_train_fold)
            dvalid = xgb.DMatrix(X_val_t, label=y_val_fold)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=50,  # smaller to prevent overfitting
                verbose_eval=False,
            )

            preds = model.predict(dvalid)
            scores.append(accuracy_10pct(y_val_fold, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=1800)

    # ---- Final Training ----
    best_params = study.best_params
    num_boost_round = best_params.pop("n_estimators")
    best_params.update({"objective": "reg:squarederror", "seed": 42, "verbosity": 0})

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat_oh", OneHotEncoder(handle_unknown="ignore"), categorical_features_oh),
        ("target_enc", TargetEncoder(smoothing=0.3, min_samples_leaf=20), target_encoded_features)
    ])

    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    dtrain_final = xgb.DMatrix(X_train_proc, label=y_train)
    dtest_final = xgb.DMatrix(X_test_proc, label=y_test)

    final_model = xgb.train(
        best_params,
        dtrain_final,
        num_boost_round=num_boost_round,
        evals=[(dtest_final, "test")],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # ---- Evaluation ----
    train_preds = final_model.predict(dtrain_final)
    test_preds = final_model.predict(dtest_final)

    train_metrics = evaluate_model(y_train, train_preds)
    test_metrics = evaluate_model(y_test, test_preds)

    print("\nTraining Metrics:")
    print(pd.DataFrame([train_metrics]))
    print("\nTest Metrics:")
    print(pd.DataFrame([test_metrics]))

    # ---- Feature Importance ----
    importance = final_model.get_score(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    }).sort_values("importance", ascending=False)

    print("\n🔎 Top 15 Feature Importances:")
    print(importance_df.head(15))

    plt.figure(figsize=(10,6))
    importance_df.head(15).plot(kind="barh", x="feature", y="importance", legend=False)
    plt.title(f"Feature Importance for {type_name}")
    plt.gca().invert_yaxis()
    plt.show()

    # ---- Save model ----
    from xgboost import XGBRegressor
    sklearn_model = XGBRegressor(**best_params, n_estimators=num_boost_round)
    sklearn_model.fit(X_train_proc, y_train)

    filename = f"real_estate_model_{type_name.replace(' ', '_')}.joblib"
    joblib.dump({
        'preprocessor': preprocessor,
        'model': sklearn_model,
        'feature_names': list(X.columns),
        'target_name': 'price_log',
        'type': type_name
    }, filename)

    print(f"✅ Model saved for {type_name} as {filename}")

    return {
        "type": type_name,
        "train_accuracy_10%": train_metrics["accuracy_10%"],
        "test_accuracy_10%": test_metrics["accuracy_10%"],
        "test_accuracy_15%": test_metrics["accuracy_15%"],
        "test_accuracy_20%": test_metrics["accuracy_20%"],
        "test_mape": test_metrics["mape"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }

# ---------------- 7. Train Models per Type ----------------
results = []
for type_group in df["type"].unique():
    df_sub = df[df["type"] == type_group].copy()
    if len(df_sub) == 0:
        continue
    metrics = train_model_for_type(df_sub, type_group)
    results.append(metrics)

# ---- Final Summary ----
results_df = pd.DataFrame(results)
print("\n📊 Accuracy Summary per Model:")
print(results_df)
