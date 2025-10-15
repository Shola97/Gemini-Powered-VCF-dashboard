import pandas as pd

df = pd.read_csv(r"C:\Users\asoku\OneDrive\Documents\OneDrive\Desktop\Projects\Data-wrangling\combined_tidy_vcf.csv")

print(df["RPB"].unique())

print(df["RPB"].dtype)