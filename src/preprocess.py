import pandas as pd
import os


def preprocess_data(input_path, output_path):
    print("📥 Loading dataset...")
    df = pd.read_csv(input_path)

    print("Original Shape:", df.shape)

    # -----------------------------
    # STEP 1: Convert wide → long
    # -----------------------------
    print("🔄 Converting to time-series format...")

    df_long = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date",
        value_name="Cases"
    )

    # -----------------------------
    # STEP 2: Clean columns
    # -----------------------------
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    df_long["Cases"] = pd.to_numeric(df_long["Cases"], errors="coerce")

    # Fill missing values
    df_long["Province/State"] = df_long["Province/State"].fillna("Unknown")
    df_long["Cases"] = df_long["Cases"].fillna(0)

    # -----------------------------
    # STEP 3: Aggregate by Country
    # -----------------------------
    print("🌍 Aggregating province-level data to country level...")

    df_country = df_long.groupby(
        ["Country/Region", "Date"],
        as_index=False
    ).agg({
        "Cases": "sum",
        "Lat": "mean",   # average location
        "Long": "mean"
    })

    # -----------------------------
    # STEP 4: Sort values
    # -----------------------------
    df_country = df_country.sort_values(
        by=["Country/Region", "Date"]
    ).reset_index(drop=True)

    # -----------------------------
    # STEP 5: Basic sanity checks
    # -----------------------------
    print("\n📊 Processed Data Info:")
    print(df_country.head())
    print(df_country.info())

    # -----------------------------
    # STEP 6: Save processed file
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_country.to_csv(output_path, index=False)

    print(f"\n✅ Processed data saved at: {output_path}")
    print("Final Shape:", df_country.shape)


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    input_file = "data/raw/raw_data.csv"   # <-- change this filename
    output_file = "data/processed/processed_data.csv"

    preprocess_data(input_file, output_file)