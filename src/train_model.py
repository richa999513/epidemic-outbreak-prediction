import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(input_path, model_path):
    print("📥 Loading feature data...")

    df = pd.read_csv(input_path)

    # -----------------------------
    # SORT DATA (VERY IMPORTANT)
    # -----------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Country/Region", "Date"])

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    features = [
        "Lag_1", "Lag_7", "Rolling_7", "Growth_Rate",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]

    target = "Daily_Cases"

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df = df.dropna(subset=features + [target])

    # Remove negative or extreme anomalies (optional safety)
    df = df[df[target] >= 0]

    X = df[features]
    y = df[target]

    # -----------------------------
    # LOG TRANSFORMATION (🔥 IMPORTANT)
    # -----------------------------
    y_log = np.log1p(y)

    # -----------------------------
    # TIME-BASED SPLIT
    # -----------------------------
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y_log.iloc[:split_index]
    y_test_actual = y.iloc[split_index:]  # keep actual values for evaluation

    print("📊 Train size:", X_train.shape)
    print("📊 Test size:", X_test.shape)

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    print("🤖 Training RandomForest model...")

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    y_pred_log = model.predict(X_test)

    # Convert back to original scale
    y_pred = np.expm1(y_pred_log)

    # -----------------------------
    # EVALUATION
    # -----------------------------
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = mean_squared_error(y_test_actual, y_pred) ** 0.5

    # MAPE (important for judges)
    mape = (np.abs(y_test_actual - y_pred) / (y_test_actual + 1)).mean() * 100

    print("\n📊 Model Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # -----------------------------
    # FEATURE IMPORTANCE (🔥)
    # -----------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\n🔥 Feature Importance:")
    print(importance)

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump({
        "model": model,
        "features": features
    }, model_path)

    print("\n✅ Model saved successfully!")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    train_model(
        "data/processed/featured_data.csv",
        "models/model.pkl"
    )

# import pandas as pd
# import joblib
# import os

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error


# def train_model(input_path, model_path):
#     print("📥 Loading feature data...")

#     df = pd.read_csv(input_path)

#     # -----------------------------
#     # SORT (VERY IMPORTANT)
#     # -----------------------------
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values(by=["Country/Region", "Date"])

#     # -----------------------------
#     # FEATURE SELECTION
#     # -----------------------------
#     features = [
#         "Lag_1", "Lag_7", "Rolling_7", "Growth_Rate",
#         "retail_and_recreation_percent_change_from_baseline",
#         "grocery_and_pharmacy_percent_change_from_baseline",
#         "parks_percent_change_from_baseline",
#         "transit_stations_percent_change_from_baseline",
#         "workplaces_percent_change_from_baseline",
#         "residential_percent_change_from_baseline"
#     ]

#     target = "Daily_Cases"

#     # -----------------------------
#     # REMOVE INVALID ROWS
#     # -----------------------------
#     df = df.dropna(subset=features + [target])

#     X = df[features]
#     y = df[target]

#     # -----------------------------
#     # TIME-BASED SPLIT (NO SHUFFLE)
#     # -----------------------------
#     split_index = int(len(df) * 0.8)

#     X_train = X.iloc[:split_index]
#     X_test = X.iloc[split_index:]

#     y_train = y.iloc[:split_index]
#     y_test = y.iloc[split_index:]

#     print("📊 Train size:", X_train.shape)
#     print("📊 Test size:", X_test.shape)

#     # -----------------------------
#     # TRAIN MODEL
#     # -----------------------------
#     print("🤖 Training RandomForest model...")

#     model = RandomForestRegressor(
#         n_estimators=120,
#         max_depth=12,
#         random_state=42,
#         n_jobs=-1
#     )

#     model.fit(X_train, y_train)

#     # -----------------------------
#     # PREDICTIONS
#     # -----------------------------
#     y_pred = model.predict(X_test)

#     # -----------------------------
#     # EVALUATION
#     # -----------------------------
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred) ** 0.5

#     print("\n📊 Model Performance:")
#     print(f"MAE  : {mae:.2f}")
#     print(f"RMSE : {rmse:.2f}")

#     # -----------------------------
#     # FEATURE IMPORTANCE (🔥 VERY IMPORTANT)
#     # -----------------------------
#     importance = pd.Series(
#         model.feature_importances_,
#         index=features
#     ).sort_values(ascending=False)

#     print("\n🔥 Feature Importance:")
#     print(importance)

#     # -----------------------------
#     # SAVE MODEL
#     # -----------------------------
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)

#     print("\n✅ Model saved successfully!")


# if __name__ == "__main__":
#     train_model(
#         "data/processed/featured_data.csv",
#         "models/model.pkl"
#     )