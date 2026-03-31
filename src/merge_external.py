import pandas as pd
import country_converter as coco
import os
import logging

# Silence country_converter warnings
logging.getLogger("country_converter").setLevel(logging.ERROR)


# -----------------------------
# REMOVE INVALID REGIONS
# -----------------------------
def remove_invalid_regions(df, column_name):
    invalid_regions = [
        "Diamond Princess",
        "MS Zaandam",
        "Summer Olympics 2020"
    ]
    return df[~df[column_name].isin(invalid_regions)]


# -----------------------------
# FALLBACK MAPPING (EDGE CASES)
# -----------------------------
def apply_fallback_mapping(df, column_name):
    fallback_mapping = {
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Congo (Brazzaville)": "Congo",
        "Ivory Coast": "Cote d'Ivoire",
        "Korea, South": "South Korea",
        "Korea, North": "North Korea",
        "US": "United States",
        "UK": "United Kingdom",
        "Taiwan*": "Taiwan",
        "Russia": "Russian Federation"
    }
    df[column_name] = df[column_name].replace(fallback_mapping)
    return df


# -----------------------------
# STANDARDIZE COUNTRY NAMES
# -----------------------------
def standardize_country_names(df, column_name):
    print(f"⚡ Optimizing country conversion for {column_name}...")

    # Get unique country names
    unique_countries = df[column_name].dropna().unique()

    # Convert only unique values
    converted = coco.convert(
        names=unique_countries,
        to='name_short',
        not_found=None
    )

    # Create mapping dictionary
    mapping = dict(zip(unique_countries, converted))

    # Apply mapping
    df[column_name] = df[column_name].map(mapping)

    return df


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def merge_mobility(cases_path, mobility_path, output_path):
    print("📥 Loading datasets...")

    cases = pd.read_csv(cases_path)
    mobility = pd.read_csv(mobility_path)

    # -----------------------------
    # STEP 1: Remove invalid regions FIRST
    # -----------------------------
    cases = remove_invalid_regions(cases, "Country/Region")

    # -----------------------------
    # STEP 2: Apply fallback mapping
    # -----------------------------
    cases = apply_fallback_mapping(cases, "Country/Region")
    mobility = apply_fallback_mapping(mobility, "country_region")

    # -----------------------------
    # STEP 3: Standardize names
    # -----------------------------
    print("🌍 Standardizing country names...")

    cases = standardize_country_names(cases, "Country/Region")
    mobility = standardize_country_names(mobility, "country_region")

    # Drop rows where conversion failed
    cases = cases.dropna(subset=["Country/Region"])
    mobility = mobility.dropna(subset=["country_region"])

    # -----------------------------
    # STEP 4: Date cleaning
    # -----------------------------
    print("📅 Processing dates...")

    cases["Date"] = pd.to_datetime(cases["Date"], errors="coerce")
    mobility["date"] = pd.to_datetime(mobility["date"], dayfirst=True, errors="coerce")

    cases = cases.dropna(subset=["Date"])
    mobility = mobility.dropna(subset=["date"])

    # -----------------------------
    # STEP 5: Keep country-level mobility only
    # -----------------------------
    mobility = mobility[mobility["sub_region_1"].isna()]

    # -----------------------------
    # STEP 6: Select columns
    # -----------------------------
    mobility = mobility[[
        "country_region",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]]

    mobility.rename(columns={
        "country_region": "Country/Region",
        "date": "Date"
    }, inplace=True)

    # -----------------------------
    # STEP 7: Merge
    # -----------------------------
    print("🔗 Merging datasets...")

    merged = pd.merge(
        cases,
        mobility,
        on=["Country/Region", "Date"],
        how="left"
    )

    # -----------------------------
    # STEP 8: Fill missing mobility values
    # -----------------------------
    mobility_cols = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]

    merged[mobility_cols] = merged[mobility_cols].fillna(0)

    # -----------------------------
    # STEP 9: Debug unmatched countries
    # -----------------------------
    unmatched = merged[
        merged[mobility_cols].sum(axis=1) == 0
    ]["Country/Region"].unique()

    print(f"⚠️ Unmatched countries (sample): {unmatched[:10]}")
    print(f"Total unmatched countries: {len(unmatched)}")

    # -----------------------------
    # STEP 10: Save
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("✅ Merge completed successfully!")
    print("Final Shape:", merged.shape)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    merge_mobility(
        "data/processed/processed_data.csv",
        "data/raw/mobility.csv",
        "data/processed/merged_data.csv"
    )