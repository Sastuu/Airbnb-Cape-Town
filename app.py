import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# -----------------------------------------------------------------------------
# Page configuration & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cape Town Superhost Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .small-note { color: #555; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Paths & model loading
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "superhost_pipeline.joblib"

@st.cache_resource
def load_pipeline(path: Path):
    return joblib.load(path)

pipeline = None
model_load_error = None
if MODEL_PATH.exists():
    try:
        pipeline = load_pipeline(MODEL_PATH)
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = (
        f"Model file not found at `{MODEL_PATH}`. "
        "Train and save your Superhost pipeline in the notebook first (L1–L3)."
    )

# -----------------------------------------------------------------------------
# App schema (MUST match the features used to train the saved pipeline)
# Update these choices to exactly match your final selected app features.
# -----------------------------------------------------------------------------
# IMPORTANT: Keep this in sync with APP_FEATURES in notebooks/01_airbnb_capstone.ipynb (L1/L3).
# The DataFrame passed to the pipeline must have the same column names and dtypes as training.
APP_FEATURES = [
    "neighbourhood_cleansed",
    "room_type",
    "accommodates",
    "bedrooms",
    "bathrooms_count",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "availability_365",
    "amenities_count",
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

# A small starter list; replace with the full set from your processed data/artifact.
NEIGHBOURHOODS = [
    "Ward 1", "Ward 4", "Ward 23", "Ward 54", "Ward 55", "Ward 57",
    "Ward 64", "Ward 74", "Ward 77", "Ward 115", "Other",
]

def build_input_row(inputs: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with the exact app schema."""
    row = {col: inputs.get(col) for col in APP_FEATURES}
    return pd.DataFrame([row], columns=APP_FEATURES)

def predict_superhost_proba(model, X: pd.DataFrame) -> float:
    proba = model.predict_proba(X)
    # Assumes positive class is encoded as 1; confirm in your notebook.
    if proba.shape[1] == 2:
        return float(proba[0, 1])
    return float(proba[0].max())

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Module",
    [
        "Superhost Predictor",
        "Batch Scoring",
        "Insights (sample)",
        "Model & App Notes",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Decision threshold")
threshold = st.sidebar.slider(
    "Flag as likely Superhost if probability ≥",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Choose and justify this in the notebook based on business needs (J4).",
)

st.sidebar.caption(
    "Data: Inside Airbnb, Cape Town snapshot 2026-09-23 (CC BY 4.0). "
    "This app is an educational capstone project."
)

# -----------------------------------------------------------------------------
# Module 1: Single prediction
# -----------------------------------------------------------------------------
if page == "Superhost Predictor":
    st.title("Superhost Probability Predictor")
    st.markdown(
        "Estimate the probability that a listing's host is a **Superhost** based on listing and host-related features. "
        "Adjust inputs to explore how the model responds."
    )

    if pipeline is None:
        st.warning(f"Model not ready: {model_load_error}")
        st.info(
            "Once `models/superhost_pipeline.joblib` exists, this page will run predictions. "
            "You can still design and test the input form now."
        )

    with st.expander("Listing profile inputs", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Location & type")
            neighbourhood = st.selectbox("Neighbourhood (cleansed)", NEIGHBOURHOODS, index=0)
            room_type = st.selectbox("Room type", ROOM_TYPES, index=0)
            amenities_count = st.number_input("Amenities count", min_value=0, max_value=200, value=20, step=1,
                                             help="Engineered feature: count of amenities (must match notebook definition).")
        with c2:
            st.subheader("Capacity")
            accommodates = st.number_input("Accommodates (guests)", min_value=1, max_value=16, value=2, step=1)
            bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
            bathrooms_count = st.number_input("Bathrooms (count)", min_value=0.0, max_value=20.0, value=1.0, step=0.5,
                                              help="Parsed bathroom count (must match notebook parsing).")
        with c3:
            st.subheader("Stay, reviews & availability")
            minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=2, step=1)
            number_of_reviews = st.number_input("Number of reviews", min_value=0, max_value=5000, value=10, step=1)
            review_scores_rating = st.number_input(
                "Review score rating", min_value=0.0, max_value=5.0, value=4.8, step=0.01,
                help="Keep missing-value handling consistent with training/preprocessing."
            )
            availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=180, step=1)

    inputs = {
        "neighbourhood_cleansed": neighbourhood,
        "room_type": room_type,
        "accommodates": int(accommodates),
        "bedrooms": float(bedrooms),
        "bathrooms_count": float(bathrooms_count),
        "minimum_nights": int(minimum_nights),
        "number_of_reviews": int(number_of_reviews),
        "review_scores_rating": float(review_scores_rating),
        "availability_365": int(availability_365),
        "amenities_count": int(amenities_count),
    }
    X_input = build_input_row(inputs)

    st.markdown("---")
    if st.button("Predict Superhost probability", type="primary", disabled=(pipeline is None)):
        try:
            proba = predict_superhost_proba(pipeline, X_input)
            pred_label = "Likely Superhost" if proba >= threshold else "Not flagged as Superhost"
            left, right = st.columns([1, 1.4])

            with left:
                st.subheader("Prediction")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    title={"text": "Superhost probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2b6cb0"},
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#edf2f7"},
                            {"range": [threshold * 100, 100], "color": "#c6f6d5"},
                        ],
                        "threshold": {
                            "line": {"color": "#e53e3e", "width": 4},
                            "thickness": 0.75,
                            "value": threshold * 100,
                        },
                    },
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("Result & interpretation")
                if proba >= threshold:
                    st.success(f"**{pred_label}** (p = {proba:.1%}, threshold = {threshold:.0%})")
                else:
                    st.info(f"**{pred_label}** (p = {proba:.1%}, threshold = {threshold:.0%})")

                st.markdown("**Input snapshot**")
                st.dataframe(X_input, use_container_width=True)

                st.markdown("**What this means (plain language)**")
                st.write(
                    "This score estimates how similar this listing's profile is to hosts labeled as Superhosts in the training data. "
                    "It is a probability from a model, not a guarantee or an official Airbnb decision."
                )

                st.markdown("**Limitations**")
                st.caption(
                    "The model is trained on a single Cape Town snapshot (2026-09-23) and only on the features included in this app. "
                    "It cannot see photos, listing text, guest messages, or changes over time, and it may perform differently across areas or listing types."
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption(
                "Common causes: feature names/order/dtypes don't match training, or the saved pipeline expects different preprocessing. "
                "Compare `X_input.columns` and dtypes to your training data."
            )

# -----------------------------------------------------------------------------
# Module 2: Batch scoring
# -----------------------------------------------------------------------------
elif page == "Batch Scoring":
    st.title("Batch Superhost Scoring")
    st.markdown(
        "Upload a CSV with the same columns used by the app/pipeline to score many listings at once."
    )

    template = pd.DataFrame([{c: "" for c in APP_FEATURES}])
    st.download_button(
        "Download CSV template (app schema)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="superhost_batch_template.csv",
        mime="text/csv",
    )

    if pipeline is None:
        st.warning(f"Model not ready: {model_load_error}")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df_in):,}** rows.")
            st.dataframe(df_in.head(), use_container_width=True)

            missing = [c for c in APP_FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif st.button("Score batch", disabled=(pipeline is None)):
                X_batch = df_in[APP_FEATURES].copy()
                probas = pipeline.predict_proba(X_batch)
                p = probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)
                out = df_in.copy()
                out["superhost_probability"] = np.round(p, 4)
                out["flagged_likely_superhost"] = (out["superhost_probability"] >= threshold)

                k1, k2, k3 = st.columns(3)
                k1.metric("Rows scored", f"{len(out):,}")
                k2.metric("Flagged likely Superhost", f"{int(out['flagged_likely_superhost'].sum()):,}")
                k3.metric("Mean probability", f"{out['superhost_probability'].mean():.1%}")

                st.dataframe(out.head(20), use_container_width=True)
                st.download_button(
                    "Download scored results",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="scored_superhost_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# -----------------------------------------------------------------------------
# Module 3: Sample insights (illustrative; replace/extend with your findings)
# -----------------------------------------------------------------------------
elif page == "Insights (sample)":
    st.title("Sample Insights")
    st.markdown(
        "This page is a **starter visualization area**. Replace or augment it with charts from your EDA and model results "
        "(saved figures or summary tables) so the app tells a richer story."
    )

    # Synthetic illustrative data so the page works before a processed dataset is wired in.
    rng = np.random.default_rng(42)
    sample = pd.DataFrame({
        "room_type": rng.choice(ROOM_TYPES, size=400, p=[0.55, 0.35, 0.05, 0.05]),
        "accommodates": rng.integers(1, 9, size=400),
        "review_scores_rating": np.clip(rng.normal(4.7, 0.3, size=400), 1, 5),
        "number_of_reviews": rng.integers(0, 200, size=400),
        "superhost_probability": np.clip(rng.beta(2, 3, size=400), 0, 1),
    })
    sample["flag"] = np.where(sample["superhost_probability"] >= threshold, "Above threshold", "Below threshold")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            sample, x="superhost_probability", nbins=20, color="flag",
            title="Example probability distribution (replace with real scored data)",
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_box = px.box(
            sample, x="room_type", y="superhost_probability", color="room_type",
            title="Example probability by room type (replace with real data)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.caption(
        "Tip: export key aggregates from your notebook (e.g. Superhost rate by room type/neighbourhood, model metrics) "
        "to a small CSV in the repo and load it here for trustworthy, richer insights."
    )

# -----------------------------------------------------------------------------
# Module 4: Model notes
# -----------------------------------------------------------------------------
elif page == "Model & App Notes":
    st.title("Model & App Notes")
    st.markdown("Use this page to document what the deployed model is, how to use it, and what to trust it for.")

    t1, t2, t3 = st.tabs(["Pipeline", "App schema", "About & credit"])

    with t1:
        st.subheader("Loaded pipeline")
        if pipeline is not None:
            st.code(repr(pipeline), language="python")
            st.caption("If your pipeline exposes feature names/importances/coefficients, you can visualize them here (see app_v2 example).")
        else:
            st.warning(model_load_error)

    with t2:
        st.subheader("Expected input features")
        st.write("The app currently expects these columns (edit `APP_FEATURES` to match your final trained pipeline):")
        st.code("\n".join(APP_FEATURES), language="text")
        st.markdown(
            "- **Column names & types must match training** (same encoding/categories handling inside the pipeline).\n"
            "- Train the final deployed pipeline on **only** these app features so users never need to provide hidden columns.\n"
            "- Record why you chose these features in your decision log."
        )

    with t3:
        st.subheader("About this project")
        st.markdown(
            """
**Business question (Task C):** Which hosts are Superhosts?  
This app shows a model-estimated Superhost probability for a listing described by the inputs above.

**Limitations (example—edit to reflect your findings):**  
- Based on one snapshot of Cape Town listings (23 Sep 2026); patterns may change over time.  
- Uses only selected structured fields; it does not use photos, descriptions, or messages.  
- Performance can differ by neighbourhood, room type, or host portfolio size (report subgroup results in your notebook/README).

**Credit:** Data from [Inside Airbnb](http://insideairbnb.com/get-the-data/) (Cape Town, 2026-09-23 snapshot), licensed under **CC BY 4.0**.
"""
        )

st.markdown("---")
st.caption("Cape Town Superhost Intelligence — capstone starter app. Credit: Inside Airbnb (CC BY 4.0).")
