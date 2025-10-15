# app_gradio.py  (VCF-quality oriented)
import os, json, pandas as pd, numpy as np, gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

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
    """Loader that matches your requested data-handling semantics.
    - Treats '', ' ', '.', and NA-like tokens as missing
    - Ensures bias columns exist & are numeric
    - Parses DP4 into DP/ALT_DEPTH/ALT_FRAC
    - Converts INDEL booleans to 1/0
    - Returns tidy columns; missing values remain None-compatible for JSON/LLM
    """
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

    # Parse DP4 column into components + totals
    if "DP4" in df.columns:
        parsed = df["DP4"].apply(_parse_dp4).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)
    else:
        for k in ["DP", "ALT_DEPTH", "ALT_FRAC"]:
            df[k] = np.nan

    # Convert INDEL to numeric indicator
    if "INDEL" in df.columns:
        df["INDEL"] = df["INDEL"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).fillna(np.nan)

    # Reorder columns to a consistent view
    preferred = [
        "CHROM","POS","REF","ALT","DP","INDEL","ALT_FRAC","MQ",
        "AC","AN","DP4","gt_GT","gt_GT_alleles","QUAL",
        "VDB","RPB","MQB","BQB","MQSB","SGB","MQ0F","gt_PL"
    ]
    df = df[[c for c in preferred if c in df.columns]]

    # Keep JSON-null friendly missing values
    df = df.where(pd.notnull(df), None)
    return df


SYSTEM_PROMPT = (
    "You are an evidence summariser for genomic variant outputs. Provide informational explanations only; "
    "do not give medical advice. Explain fields in plain English (DP/DP4, ALT_FRAC, MQ, and bias metrics). "
    "If a field is null/missing, say 'not computed' and do not infer values."
)


def normalise_for_json(df: pd.DataFrame):
    """Ensure NaN -> None so JSON contains nulls."""
    return json.loads(df.where(pd.notnull(df), None).to_json(orient="records"))


def explain_variant(row_json_or_dict):
    # Accept either a JSON string or a dict
    try:
        row = json.loads(row_json_or_dict) if isinstance(row_json_or_dict, str) else row_json_or_dict
    except Exception:
        return "Please paste a valid JSON row (use the table's copy-as-JSON)."

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(row)}
    ]
    try:
        resp = LLM.invoke(msgs)
        return str(resp.content)
    except Exception as e:
        return f"LLM error: {e}"


def filter_table(df: pd.DataFrame, min_mq, min_dp, indel_only):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Apply numeric filters safely (treat None as 0 for thresholds)
    if min_mq is not None and "MQ" in out.columns:
        out = out[pd.to_numeric(out["MQ"], errors="coerce").fillna(0) >= float(min_mq)]
    if min_dp is not None and "DP" in out.columns:
        out = out[pd.to_numeric(out["DP"], errors="coerce").fillna(0) >= float(min_dp)]

    if bool(indel_only) and "INDEL" in out.columns:
        # Keep only rows where INDEL == 1
        out = out[pd.to_numeric(out["INDEL"], errors="coerce").fillna(0) == 1]

    # Maintain JSON-friendly missing
    out = out.where(pd.notnull(out), None)
    return out.reset_index(drop=True)


with gr.Blocks(title="VCF Helper (Informational only)") as demo:
    gr.Markdown("### Variant Helper (CSV from VCF)\n**Informational only — not medical advice.**")

    file = gr.File(label="Upload tidy VCF-derived CSV")

    with gr.Row():
        min_mq = gr.Number(label="Min MQ (e.g., 40)", value=None)
        min_dp = gr.Number(label="Min DP (e.g., 10)", value=None)
        indel_only = gr.Checkbox(label="INDEL only", value=False)

    btn_load = gr.Button("Load / Refresh")
    table = gr.Dataframe(row_count=(8, "dynamic"), wrap=True, interactive=False)
    btn_apply = gr.Button("Apply filters")

    gr.Markdown("**Explain a row:** copy a row as JSON from the table and paste below.")
    row_json = gr.Textbox(label="Selected row (JSON)", lines=6)
    btn_explain = gr.Button("Explain with Gemini")
    explanation = gr.Markdown()

    df_state = gr.State(pd.DataFrame())

    def do_load(file):
        if not file:
            return pd.DataFrame()
        return load_csv(file)

    btn_load.click(fn=do_load, inputs=file, outputs=table).then(
        lambda t: t, inputs=table, outputs=df_state
    )

    def do_filter(df, min_mq, min_dp, indel_only):
        return filter_table(df, min_mq, min_dp, indel_only)

    btn_apply.click(fn=do_filter, inputs=[df_state, min_mq, min_dp, indel_only], outputs=table)
    btn_explain.click(fn=explain_variant, inputs=row_json, outputs=explanation)


demo.launch()
