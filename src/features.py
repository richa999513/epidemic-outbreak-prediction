import pandas as pd
import numpy as np
import os


def create_features(input_path, output_path):
    print("📥 Loading processed data...")
    df = pd.read_csv(input_path)

    # -----------------------------
    # Convert Date + Sort
    # -----------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Country/Region", "Date"])

    print("⚙️ Creating features...")

    # -----------------------------
    # 1. Daily Cases
    # -----------------------------
    df["Daily_Cases"] = df.groupby("Country/Region")["Cases"].diff()
    df["Daily_Cases"] = df["Daily_Cases"].fillna(0)
    df["Daily_Cases"] = df["Daily_Cases"].clip(lower=0)

    # -----------------------------
    # 2. Lag Features FIRST (needed for growth)
    # -----------------------------
    df["Lag_1"] = df.groupby("Country/Region")["Daily_Cases"].shift(1)
    df["Lag_7"] = df.groupby("Country/Region")["Daily_Cases"].shift(7)

    df["Lag_1"] = df["Lag_1"].fillna(0)
    df["Lag_7"] = df["Lag_7"].fillna(0)

    # -----------------------------
    # 3. Growth Rate (🔥 FIXED VERSION)
    # -----------------------------
    # Use Daily Cases instead of total Cases (VERY IMPORTANT)
    df["Growth_Rate"] = (df["Daily_Cases"] - df["Lag_1"]) / (df["Lag_1"] + 10)

    # Handle extreme values
    df["Growth_Rate"] = df["Growth_Rate"].replace([np.inf, -np.inf], 0)
    df["Growth_Rate"] = df["Growth_Rate"].fillna(0)

    # Clamp values (prevents explosion)
    df["Growth_Rate"] = df["Growth_Rate"].clip(-5, 5)

    # -----------------------------
    # 4. Rolling Average (7 days)
    # -----------------------------
    df["Rolling_7"] = (
        df.groupby("Country/Region")["Daily_Cases"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

    # -----------------------------
    # 5. Risk Level (Improved)
    # -----------------------------
    def classify_risk(cases):
        if cases > 10000:
            return "HIGH"
        elif cases > 1000:
            return "MEDIUM"
        else:
            return "LOW"

    df["Risk_Level"] = df["Rolling_7"].apply(classify_risk)

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n📊 Feature Data Sample:")
    print(df.head())

    print("\n✅ Feature engineering completed!")
    print("Final Shape:", df.shape)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    input_file = "data/processed/merged_data.csv"
    output_file = "data/processed/featured_data.csv"

    create_features(input_file, output_file)