#  (VCF-quality oriented)
import os, json, pandas as pd, numpy as np, gradio as gr

EXPECTED = [
    "VDB","RPB","MQB","BQB","MQSB","SGB","MQ0F", "INDEL",
    "AC","AN","DP4","MQ","gt_PL","gt_GT","gt_GT_alleles",
    # CHROM/POS are optional; include if present in your CSV
    "CHROM","POS","REF","ALT","QUAL"
]

def _num(x):
    try: return float(x)
    except: return np.nan

def _parse_dp4(dp4):
    """DP4 like 'ref_fwd,ref_rev,alt_fwd,alt_rev' -> dict + totals"""
    if isinstance(dp4, str):
        parts = [p.strip() for p in dp4.split(",")]
    elif isinstance(dp4, (list, tuple)):
        parts = list(dp4)
    else:
        return dict(ref_fwd=np.nan, ref_rev=np.nan, alt_fwd=np.nan, alt_rev=np.nan, DP=np.nan, ALT_DEPTH=np.nan, ALT_FRAC=np.nan)
    if len(parts) != 4:
        return dict(ref_fwd=np.nan, ref_rev=np.nan, alt_fwd=np.nan, alt_rev=np.nan, DP=np.nan, ALT_DEPTH=np.nan, ALT_FRAC=np.nan)
    r1,r2,a1,a2 = [_num(p) for p in parts]
    dp = r1+r2+a1+a2
    ad_alt = a1+a2
    alt_frac = (ad_alt/dp) if dp and dp>0 else np.nan
    return dict(ref_fwd=r1, ref_rev=r2, alt_fwd=a1, alt_rev=a2, DP=dp, ALT_DEPTH=ad_alt, ALT_FRAC=alt_frac)

BIAS_COLS = ["RPB","MQB","BQB","MQSB"]

def load_csv(file):

    # Read CSV with explicit NA handling
    df = pd.read_csv(
    file.name,
    na_values=["NA", "N/A", "na", "n/a", ".", "", " "],
    keep_default_na=False
)


    # Ensure bias columns exist and are numeric
    for c in BIAS_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep expected columns only
    cols = [c for c in df.columns if c in EXPECTED]
    if cols:
        df = df[cols]

    # Parse DP4 column
    if "DP4" in df.columns:
        parsed = df["DP4"].apply(_parse_dp4).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)
    else:
        for k in ["DP", "ALT_DEPTH", "ALT_FRAC"]:
            df[k] = np.nan

    # Convert INDEL to numeric
    if "INDEL" in df.columns:
        df["INDEL"] = df["INDEL"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).fillna(np.nan)

    # Reorder columns
    preferred = [
        "CHROM","POS","REF","ALT","DP","INDEL","ALT_FRAC","MQ",
        "AC","AN","DP4","gt_GT","gt_GT_alleles","QUAL",
        "VDB","RPB","MQB","BQB","MQSB","SGB","MQ0F","gt_PL"
    ]
    df = df[[c for c in preferred if c in df.columns]]

    # Final cleanup: ensure all nulls are consistent
    df = df.where(pd.notnull(df), None)
    return df


    
if __name__ == "__main__":
    # Path to your test CSV
    csv_path = r"C:\Users\asoku\OneDrive\Documents\OneDrive\Desktop\Projects\Data-wrangling\combined_tidy_vcf.csv"  # change this to your file path

    class File:
        def __init__(self, name): self.name = name

    # Load CSV through your existing function
    df = load_csv(File(csv_path))

    # Show top 10 rows and info
    print(df.head(10))
    print(df.info())
    # Save to new CSV
    df.to_csv("processed_vcf_data.csv", index=False)

    print(df[BIAS_COLS].isna().sum())
    print(df.dtypes)
