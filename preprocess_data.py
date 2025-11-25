"""
Data preprocessing for Index Replication Project

This script preprocesses the Fama-French industry portfolio data and market factors
to comply with the project constraints:
1. Exclude industries #2 (Mining and Materials) and #3 (Oil and Petroleum Products)
2. Prepare for weight constraints: Trans (#13) ≤ 1%, Utils (#14) ≤ 1%, Machn (#11) ≤ 8%
3. Calculate total market return (Mkt-RF + RF)
4. Handle missing data
5. Align dates between datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define file paths
DATA_DIR = Path(__file__).parent / 'data'
INDUSTRY_FILE = DATA_DIR / '17_Industry_Portfolios.csv'
FACTORS_FILE = DATA_DIR / 'F-F_Research_Data_Factors.csv'
OUTPUT_DIR = DATA_DIR / 'preprocessed'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_industry_portfolios():
    """Load the 17 industry portfolios data."""
    print("Loading industry portfolios...")

    # Read the CSV, skipping header rows
    # Row 12 (index 11) contains column names, data starts at row 13 (index 12)
    df = pd.read_csv(INDUSTRY_FILE, skiprows=11)

    # Clean up column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Rename the first column to 'Date'
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert date column from YYYYMM to datetime
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m')

    # Get the column names (industries)
    industry_cols = df.columns[1:].tolist()
    print(f"Found {len(industry_cols)} industries: {industry_cols}")

    # Replace missing value indicators with NaN
    df.replace([-99.99, -999], np.nan, inplace=True)

    # Keep only monthly data (filter out annual summaries if present)
    # Monthly data has dates like YYYYMM where MM is between 01 and 12
    df = df[df['Date'].notna()]

    print(f"Loaded {len(df)} monthly observations from {df['Date'].min()} to {df['Date'].max()}")

    return df, industry_cols

def load_market_factors():
    """Load Fama-French market factors and calculate total market return."""
    print("\nLoading market factors...")

    # Read the CSV, skipping header rows
    # Row 5 (index 4) contains column names, data starts at row 6 (index 5)
    df = pd.read_csv(FACTORS_FILE, skiprows=4)

    # Clean up column names
    df.columns = df.columns.str.strip()

    # Rename the first column to 'Date'
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert date column from YYYYMM to datetime
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m')

    # Keep only rows with valid dates (filter out annual summaries)
    df = df[df['Date'].notna()]

    # Calculate total market return: Mkt-RF + RF
    df['Mkt'] = df['Mkt-RF'] + df['RF']

    print(f"Loaded {len(df)} monthly observations from {df['Date'].min()} to {df['Date'].max()}")

    return df

def merge_and_filter_data(industry_df, factors_df, industry_cols):
    """Merge industry and market data, apply constraints."""
    print("\nMerging and filtering data...")

    # Merge on Date
    merged_df = pd.merge(industry_df, factors_df[['Date', 'Mkt', 'RF']], on='Date', how='inner')
    print(f"After merge: {len(merged_df)} observations from {merged_df['Date'].min()} to {merged_df['Date'].max()}")

    # Industry mapping (1-indexed as per project description)
    industry_mapping = {i+1: col for i, col in enumerate(industry_cols)}
    print("\nIndustry mapping:")
    for idx, name in industry_mapping.items():
        print(f"  Industry #{idx}: {name}")

    # Identify industries to exclude
    # Industry #2: Mines (Mining and Materials)
    # Industry #3: Oil (Oil and Petroleum Products)
    excluded_industries = [industry_mapping[2], industry_mapping[3]]
    print(f"\nExcluding industries: #{2} ({excluded_industries[0]}), #{3} ({excluded_industries[1]})")

    # Identify industries with special weight constraints
    constrained_industries = {
        11: (industry_mapping[11], 0.08),  # Machinery and Business Equipment ≤ 8%
        13: (industry_mapping[13], 0.01),  # Transportation ≤ 1%
        14: (industry_mapping[14], 0.01),  # Utilities ≤ 1%
    }
    print("\nIndustries with special weight constraints:")
    for idx, (name, max_weight) in constrained_industries.items():
        print(f"  Industry #{idx} ({name}): max weight = {max_weight*100}%")

    # Create a clean dataframe with allowed industries only
    allowed_industries = [col for col in industry_cols if col not in excluded_industries]
    print(f"\nAllowed industries ({len(allowed_industries)}):")
    for i, col in enumerate(allowed_industries, 1):
        orig_idx = industry_cols.index(col) + 1
        print(f"  {i}. {col} (original industry #{orig_idx})")

    # Create output dataframe
    output_df = merged_df[['Date'] + allowed_industries + ['Mkt', 'RF']].copy()

    # Handle any remaining missing values
    missing_counts = output_df.isnull().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values detected:")
        print(missing_counts[missing_counts > 0])
        print("Dropping rows with missing values...")
        output_df = output_df.dropna()

    print(f"\nFinal dataset: {len(output_df)} observations")
    print(f"Date range: {output_df['Date'].min()} to {output_df['Date'].max()}")
    print(f"Number of years: {(output_df['Date'].max() - output_df['Date'].min()).days / 365.25:.1f}")

    return output_df, allowed_industries, industry_mapping

def save_preprocessed_data(df, allowed_industries):
    """Save preprocessed data and metadata."""
    print("\nSaving preprocessed data...")

    # Save main data file
    output_file = OUTPUT_DIR / 'industry_returns_preprocessed.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Save metadata/documentation
    metadata_file = OUTPUT_DIR / 'preprocessing_info.txt'
    with open(metadata_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Index Replication Project - Data Preprocessing Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONSTRAINTS APPLIED:\n")
        f.write("-" * 80 + "\n")
        f.write("1. EXCLUDED INDUSTRIES:\n")
        f.write("   - Industry #2: Mines (Mining and Materials)\n")
        f.write("   - Industry #3: Oil (Oil and Petroleum Products)\n\n")

        f.write("2. WEIGHT CONSTRAINTS (to be applied in optimization):\n")
        f.write("   - Industry #11 (Machn - Machinery and Business Equipment): max 8%\n")
        f.write("   - Industry #13 (Trans - Transportation): max 1%\n")
        f.write("   - Industry #14 (Utils - Utilities): max 1%\n")
        f.write("   - All other industries: max 20%\n")
        f.write("   - All weights must be non-negative\n")
        f.write("   - All weights must sum to 100%\n\n")

        f.write("3. TRANSACTION COSTS:\n")
        f.write("   - 0.10% of dollar amount traded (both buy and sell)\n")
        f.write("   - Portfolio must be self-financing\n\n")

        f.write("DATA SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total observations: {len(df)}\n")
        f.write(f"Date range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}\n")
        f.write(f"Number of years: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f}\n\n")

        f.write("INCLUDED INDUSTRIES (after exclusions):\n")
        f.write("-" * 80 + "\n")
        for i, ind in enumerate(allowed_industries, 1):
            f.write(f"{i:2d}. {ind}\n")

        f.write("\nCOLUMNS IN OUTPUT FILE:\n")
        f.write("-" * 80 + "\n")
        f.write("- Date: Month (YYYY-MM-DD format)\n")
        for ind in allowed_industries:
            f.write(f"- {ind}: Monthly return (%) for {ind} industry portfolio\n")
        f.write("- Mkt: Total market return (%) = Mkt-RF + RF\n")
        f.write("- RF: Risk-free rate (%)\n\n")

        f.write("NOTES:\n")
        f.write("-" * 80 + "\n")
        f.write("- All returns are in percentage points (e.g., 2.89 means 2.89%)\n")
        f.write("- Missing values (indicated by -99.99 or -999) have been removed\n")
        f.write("- Market return is calculated as Mkt-RF + RF per project instructions\n")
        f.write("- Data source: Kenneth French Data Library\n")
        f.write("- Industry portfolios are value-weighted\n")
        f.write("- Monthly rebalancing is assumed in the optimization problem\n")

    print(f"Saved: {metadata_file}")

    # Save a summary statistics file
    stats_file = OUTPUT_DIR / 'data_statistics.csv'
    stats = df[allowed_industries + ['Mkt']].describe()
    stats.to_csv(stats_file)
    print(f"Saved: {stats_file}")

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("INDEX REPLICATION PROJECT - DATA PREPROCESSING")
    print("="*80)

    # Load data
    industry_df, industry_cols = load_industry_portfolios()
    factors_df = load_market_factors()

    # Merge and filter
    preprocessed_df, allowed_industries, industry_mapping = merge_and_filter_data(
        industry_df, factors_df, industry_cols
    )

    # Save results
    save_preprocessed_data(preprocessed_df, allowed_industries)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the preprocessed data in 'industry_returns_preprocessed.csv'")
    print("2. Check 'preprocessing_info.txt' for constraint details")
    print("3. Use this data for the optimization problem with Gurobi")

if __name__ == "__main__":
    main()
