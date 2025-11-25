"""
Quick verification script to check the preprocessed data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load preprocessed data
DATA_FILE = Path(__file__).parent / 'data' / 'preprocessed' / 'industry_returns_preprocessed.csv'
df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])

print("="*80)
print("DATA VERIFICATION SUMMARY")
print("="*80)

print("\n1. DATA DIMENSIONS:")
print(f"   Total observations: {len(df)}")
print(f"   Number of industries (after exclusions): {len(df.columns) - 3}")  # Exclude Date, Mkt, RF
print(f"   Date range: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
print(f"   Time span: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f} years")

print("\n2. INDUSTRY COLUMNS:")
industry_cols = [col for col in df.columns if col not in ['Date', 'Mkt', 'RF']]
for i, col in enumerate(industry_cols, 1):
    print(f"   {i:2d}. {col}")

print("\n3. MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✓ No missing values detected!")
else:
    print("   Missing values by column:")
    for col, count in missing[missing > 0].items():
        print(f"   - {col}: {count} missing values")

print("\n4. DATA QUALITY CHECKS:")
print(f"   ✓ Mines and Oil excluded: {('Mines' not in df.columns) and ('Oil' not in df.columns)}")
print(f"   ✓ Date column is datetime: {pd.api.types.is_datetime64_any_dtype(df['Date'])}")
print(f"   ✓ All returns are numeric: {all(pd.api.types.is_numeric_dtype(df[col]) for col in industry_cols + ['Mkt', 'RF'])}")

print("\n5. SUMMARY STATISTICS (annualized %):")
# Calculate annualized statistics
monthly_returns = df[industry_cols + ['Mkt']].copy()
annualized_mean = monthly_returns.mean() * 12
annualized_std = monthly_returns.std() * np.sqrt(12)

summary = pd.DataFrame({
    'Ann. Mean': annualized_mean,
    'Ann. Std': annualized_std,
    'Sharpe': annualized_mean / annualized_std
}).round(2)

print("\n   Top 5 by Annualized Return:")
print(summary.sort_values('Ann. Mean', ascending=False).head())

print("\n   Market Statistics:")
print(f"   - Annualized Return: {annualized_mean['Mkt']:.2f}%")
print(f"   - Annualized Volatility: {annualized_std['Mkt']:.2f}%")
print(f"   - Sharpe Ratio: {(annualized_mean['Mkt'] / annualized_std['Mkt']):.2f}")

print("\n6. CONSTRAINTS READY TO APPLY:")
print("   The following constraints will be enforced during optimization:")
print("   ✓ Exclude: Mines (#2), Oil (#3)")
print("   ✓ Max weights: Machn ≤ 8%, Trans ≤ 1%, Utils ≤ 1%")
print("   ✓ All industries: max 20%, min 0%")
print("   ✓ Weights sum to 100%")
print("   ✓ Transaction cost: 0.10% per trade")

print("\n7. SAMPLE DATA (first 5 rows):")
print(df.head().to_string())

print("\n" + "="*80)
print("PREPROCESSING VERIFICATION COMPLETE!")
print("="*80)
print("\nData is ready for optimization with Gurobi.")
print(f"Load data with: pd.read_csv('{DATA_FILE}')")
