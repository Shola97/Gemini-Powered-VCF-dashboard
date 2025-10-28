
import os, json, pandas as pd, numpy as np, gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt

load_dotenv()
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

EXPECTED = [
    "VDB","RPB","MQB","BQB","MQSB","SGB","MQ0F", "INDEL",
    "AC","AN","DP4","MQ","gt_PL","gt_GT","gt_GT_alleles",
    "CHROM","POS","REF","ALT","QUAL"
]
BIAS_COLS = ["RPB","MQB","BQB","MQSB"]

def _num(x):
    try: return float(x)
    except: return np.nan

def _parse_dp4(dp4):
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

def load_csv(file):
    df = pd.read_csv(
        file.name,
        na_values=["NA", "N/A", "na", "n/a", ".", "", " "],
        keep_default_na=False
    )
    for c in BIAS_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols = [c for c in df.columns if c in EXPECTED]
    if cols:
        df = df[cols]

    if "DP4" in df.columns:
        parsed = df["DP4"].apply(_parse_dp4).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)
    else:
        for k in ["DP", "ALT_DEPTH", "ALT_FRAC"]:
            df[k] = np.nan

    if "INDEL" in df.columns:
        df["INDEL"] = df["INDEL"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).fillna(np.nan)

    preferred = [
        "CHROM","POS","REF","ALT","DP","INDEL","ALT_FRAC","MQ",
        "AC","AN","DP4","gt_GT","gt_GT_alleles","QUAL",
        "VDB","RPB","MQB","BQB","MQSB","SGB","MQ0F","gt_PL"
    ]
    df = df[[c for c in preferred if c in df.columns]]
    df = df.where(pd.notnull(df), None)
    return df

SYSTEM_PROMPT = (
    "You are an evidence summariser for genomic variant outputs. Provide informational explanations only; "
    "do not give medical advice. Explain fields in plain English (DP/DP4, ALT_FRAC, MQ, and bias metrics). "
    "If a field is null/missing, say 'not computed' and do not infer values."
)

def explain_variant(row_json_or_dict):
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

def get_row_json(df, index):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No data loaded."
    try:
        row = df.iloc[int(index)].to_dict()
        return json.dumps(row, indent=2)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="VCF Helper (Informational only)") as demo:
    gr.Markdown("### Variant Helper (CSV from VCF)\n**Informational only — not medical advice.**")

    file = gr.File(label="Upload tidy VCF-derived CSV")
    df_state = gr.State(pd.DataFrame())

    with gr.Tabs():
        with gr.Tab("Table & Explain"):
            btn_load = gr.Button("Load / Refresh")
            table = gr.Dataframe(row_count=(8, "dynamic"), wrap=True, interactive=False)

            with gr.Row():
                row_index = gr.Number(label="Row index to explain (0-based)", value=0)
                btn_preview_json = gr.Button("Preview row as JSON")

            row_json = gr.Textbox(label="Selected row (JSON)", lines=6)
            btn_explain = gr.Button("Explain with Gemini")
            status = gr.Label(value="", label="Status")
            explanation = gr.Markdown()

        with gr.Tab("Summary (Visual)"):
            gr.Markdown("Upload then click **Build visuals**. All charts auto-handle missing values.")
            btn_visual = gr.Button("Build visuals")
            with gr.Row():
                plot_dp = gr.Plot(label="Depth (DP) histogram")
                plot_mq = gr.Plot(label="Mapping Quality (MQ) histogram")
            with gr.Row():
                plot_af = gr.Plot(label="ALT_FRAC histogram")
                plot_indel = gr.Plot(label="INDEL vs SNV counts")
            with gr.Row():
                plot_rpb = gr.Plot(label="RPB distribution")
                plot_mqb = gr.Plot(label="MQB distribution")
            with gr.Row():
                plot_bqb = gr.Plot(label="BQB distribution")
                plot_mqsb = gr.Plot(label="MQSB distribution")

    # Callbacks
    def do_load(file):
        if not file:
            return pd.DataFrame(), pd.DataFrame()
        df = load_csv(file)
        df = df.reset_index().rename(columns={"index": "row_number"})
        #reorder the table with the row first
        cols = ["row_number"] + [c for c in df.columns if c != "row_number"]
        df = df[cols]
        return df, df

    btn_load.click(fn=do_load, inputs=file, outputs=[table, df_state])
    btn_preview_json.click(fn=get_row_json, inputs=[df_state, row_index], outputs=row_json)
    
    def explain_with_status(row_json):
        "shows the loading bar"
        status_msg = gr.update(value="Explaining with Gemini...")
        explanation_text = explain_variant(row_json)
        status_done = gr.update(value="Done ✅")
        return status_msg, explanation_text, status_done

    btn_explain.click(
        fn=explain_with_status,
        inputs=row_json,
        outputs=[status, explanation, status]
        )
    

    def build_visuals(df):
        # Provide default empty plots if df is empty or not a DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            fig_empty = plt.figure()
            plt.text(0.5, 0.5, "No data", ha='center', va='center')
            plt.axis('off')
            return [fig_empty]*8

        figs = []

        # DP histogram
        fig1, ax1 = plt.subplots()
        if "DP" in df.columns:
            ax1.hist(pd.to_numeric(df["DP"], errors="coerce").dropna(), bins=30, color='skyblue')
            ax1.set_title("Depth (DP)")
            ax1.set_xlabel("DP")
            ax1.set_ylabel("Count")
        else:
            ax1.text(0.5, 0.5, "DP not found", ha='center', va='center')
            ax1.axis('off')
        figs.append(fig1)

        # MQ histogram
        fig2, ax2 = plt.subplots()
        if "MQ" in df.columns:
            ax2.hist(pd.to_numeric(df["MQ"], errors="coerce").dropna(), bins=30, color='orange')
            ax2.set_title("Mapping Quality (MQ)")
            ax2.set_xlabel("MQ")
            ax2.set_ylabel("Count")
        else:
            ax2.text(0.5, 0.5, "MQ not found", ha='center', va='center')
            ax2.axis('off')
        figs.append(fig2)

        # ALT_FRAC histogram
        fig3, ax3 = plt.subplots()
        if "ALT_FRAC" in df.columns:
            ax3.hist(pd.to_numeric(df["ALT_FRAC"], errors="coerce").dropna(), bins=30, color='green')
            ax3.set_title("ALT_FRAC")
            ax3.set_xlabel("ALT_FRAC")
            ax3.set_ylabel("Count")
        else:
            ax3.text(0.5, 0.5, "ALT_FRAC not found", ha='center', va='center')
            ax3.axis('off')
        figs.append(fig3)

        # INDEL vs SNV counts
        fig4, ax4 = plt.subplots()
        if "INDEL" in df.columns:
            counts = df["INDEL"].value_counts(dropna=False)
            ax4.bar(["SNV", "INDEL"], [counts.get(0, 0), counts.get(1, 0)], color=['blue', 'red'])
            ax4.set_title("INDEL vs SNV counts")
            ax4.set_ylabel("Count")
        else:
            ax4.text(0.5, 0.5, "INDEL not found", ha='center', va='center')
            ax4.axis('off')
        figs.append(fig4)

        # RPB distribution
        fig5, ax5 = plt.subplots()
        if "RPB" in df.columns:
            ax5.hist(pd.to_numeric(df["RPB"], errors="coerce").dropna(), bins=30, color='purple')
            ax5.set_title("RPB distribution")
            ax5.set_xlabel("RPB")
            ax5.set_ylabel("Count")
        else:
            ax5.text(0.5, 0.5, "RPB not found", ha='center', va='center')
            ax5.axis('off')
        figs.append(fig5)

        # MQB distribution
        fig6, ax6 = plt.subplots()
        if "MQB" in df.columns:
            ax6.hist(pd.to_numeric(df["MQB"], errors="coerce").dropna(), bins=30, color='brown')
            ax6.set_title("MQB distribution")
            ax6.set_xlabel("MQB")
            ax6.set_ylabel("Count")
        else:
            ax6.text(0.5, 0.5, "MQB not found", ha='center', va='center')
            ax6.axis('off')
        figs.append(fig6)

        # BQB distribution
        fig7, ax7 = plt.subplots()
        if "BQB" in df.columns:
            ax7.hist(pd.to_numeric(df["BQB"], errors="coerce").dropna(), bins=30, color='pink')
            ax7.set_title("BQB distribution")
            ax7.set_xlabel("BQB")
            ax7.set_ylabel("Count")
        else:
            ax7.text(0.5, 0.5, "BQB not found", ha='center', va='center')
            ax7.axis('off')
        figs.append(fig7)

        # MQSB distribution
        fig8, ax8 = plt.subplots()
        if "MQSB" in df.columns:
            ax8.hist(pd.to_numeric(df["MQSB"], errors="coerce").dropna(), bins=30, color='gray')
            ax8.set_title("MQSB distribution")
            ax8.set_xlabel("MQSB")
            ax8.set_ylabel("Count")
        else:
            ax8.text(0.5, 0.5, "MQSB not found", ha='center', va='center')
            ax8.axis('off')
        figs.append(fig8)

        return figs

    btn_visual.click(
        fn=build_visuals,
        inputs=df_state,
        outputs=[plot_dp, plot_mq, plot_af, plot_indel, plot_rpb, plot_mqb, plot_bqb, plot_mqsb]
    )

demo.launch(share=True)  # script has to be running for the share link to work

