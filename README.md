# Airbnb Cape Town Superhost Prediction (Capstone)

A machine learning capstone project that predicts whether an Airbnb host in Cape Town is a **Superhost** (Task C, deployed app) and models nightly **price** (Task R, notebook only), using the Inside Airbnb Cape Town snapshot.

## Live App
- **Streamlit app URL:** [ADD YOUR DEPLOYED URL HERE] <!-- Not deployed yet; run locally with `streamlit run app.py` -->

## Business Questions
1. **Task C (Classification):** Which hosts are Superhosts?  
   *Who uses it / decision:* An Airbnb host-support or host-success team in Cape Town can use this prediction to prioritize outreach. The model estimates the probability that a listing’s host is a Superhost based on listing features, helping the team flag likely Superhosts for recognition or identify near-miss hosts for coaching. It is decision support, not an official Airbnb Superhost decision.
2. **Task R (Regression):** What nightly price should a listing charge?  
   *Who uses it / decision:* A host, property manager, or analyst can use the price estimate as a sanity check when setting or reviewing nightly prices. It helps compare a listing’s features (location, room type, capacity, reviews, availability, etc.) against similar Cape Town listings, while recognizing that real pricing also depends on seasonality, photos, and demand.

## Dataset
- **Source:** [Inside Airbnb - Get the Data](http://insideairbnb.com/get-the-data/) (Cape Town, South Africa)
- **Snapshot:** 23 September 2026
- **Main file:** `listings.csv.gz` (27,760 rows, 90 columns)
- **Neighbourhood file:** `neighbourhoods.csv`
- **Direct downloads:**
  - Listings: https://data.insideairbnb.com/south-africa/wc/cape-town/2026-09-23/data/listings.csv.gz
  - Neighbourhoods: https://data.insideairbnb.com/south-africa/wc/cape-town/2026-09-23/visualisations/neighbourhoods.csv
- **Credit:** Data from Inside Airbnb (http://insideairbnb.com), Cape Town snapshot dated 23 September 2026, licensed under **CC BY 4.0**.

> Raw data is not included in this repository. Download the files above and place them in `data/raw/`.

## Key Findings (EDA)
1. **What the data contains:** The Cape Town Inside Airbnb snapshot (`listings.csv.gz`, 23 Sep 2026) has **27,760 rows and 90 columns** with numeric, categorical, text, and badly typed fields. Key targets are `host_is_superhost` (t/f) and `price` (currency string).
2. **What stands out:** Superhost is imbalanced (more non-Superhosts than Superhosts), and `price` is strongly right-skewed with a long high-price tail. Review scores cluster near high values, while reviews/minimum nights/availability have long tails.
3. **What needs cleaning:** `price` must be parsed from `$`/comma text to numeric ZAR; `bathrooms_text` must be parsed into a numeric count (including half-baths). Many columns are IDs/URLs/free text/empty and need documented drop/keep decisions; missing/invalid prices and extreme rows require inspection.
4. **What is risky for modelling:** Hosts can own multiple listings, so random splits can leak host behavior; a **group-aware split by `host_id`** is safer. Some host-performance fields can leak Superhost status, and price outliers/skew can distort regression errors.
5. **What I plan to try / takeaway:** Use train-only preprocessing in pipelines, GroupKFold/group split, task-appropriate metrics (F1/precision/recall/ROC-AUC for Task C; RMSE/MAE for Task R), compare baseline/linear/tree/ensemble/XGBoost models, and test engineered features like `amenities_count`.

## Models and Results

### Task C - Superhost Classification (deployed in app)
- **Target:** `host_is_superhost` (`t/f` -> `1/0`)
- **Final model in app:** `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=neg/pos, eval_metric='logloss', random_state=42)` in a pipeline with median imputation for numeric features and most-frequent + one-hot encoding for categorical features. This matches the best validation F1/ROC-AUC model family in H8 and is saved as `models/superhost_pipeline.joblib`.
- **App features:** `neighbourhood_cleansed, room_type, accommodates, bedrooms, bathrooms_count, minimum_nights, number_of_reviews, review_scores_rating, availability_365, amenities_count`
- **Metrics chosen:** Accuracy, Precision, Recall, F1, ROC-AUC  
  *Why:* Accuracy alone can look acceptable on imbalanced Superhost labels while missing real Superhosts. Precision matters to avoid wasting support-team time on false alarms; recall matters to avoid missing true Superhosts; F1 balances both; ROC-AUC measures ranking quality across thresholds.
- **Final test scores (app-feature XGBoost pipeline, same group-aware split, run once):**
  - Accuracy: 0.7314
  - Precision: 0.5894
  - Recall: 0.6698
  - F1: 0.6271
  - ROC-AUC: 0.7862
- **Chosen threshold:** 0.50 (review against support-team capacity; retune threshold on validation for the XGBoost model before final submission)

### Task R - Price Regression (notebook only)
- **Target:** `price` parsed from currency text (e.g. `$1,114.00`) to numeric ZAR (`price_clean`); blank/invalid/non-positive prices treated as invalid for Task R and excluded from Task R modeling/evaluation.
- **Final model:** Selected by notebook J1 validation RMSE among fitted regression candidates (see notebook for exact selected model). Test metrics below are from the J1 final test output.
- **Metrics chosen:** RMSE and MAE (ZAR)  
  *Why:* MAE gives an easy-to-read average rand error; RMSE penalizes large mistakes more heavily, which matters for pricing because a few very bad price predictions can be costly.
- **Final test scores (run once, from notebook J1):**
  - RMSE (ZAR): 4319.47
  - MAE (ZAR): 1991.22

## Where the Models Fail
- **Task C - confident false positives:** Listings with strong review/quality signals can be predicted as Superhost even when the host does not meet all Superhost criteria, because those criteria are not fully represented in structured features.
- **Task C - missed Superhosts / threshold tradeoff:** Even with XGBoost, some true Superhosts have modest/typical listing features and are hard to distinguish. Threshold choice changes the precision/recall balance; tune it on validation for support-team capacity before final use.
- **Task R - luxury/rare listings:** High-price or unique properties are underpredicted/overpredicted because they are sparse and have extreme values not well represented by simple structured features.
- **Task R - missing context:** Price predictions miss important drivers not in the model (seasonality, photos, listing text quality, exact micro-location, demand/events), leading to large residuals.
- **Subgroup/performance limitation:** Performance can vary by room type, neighbourhood, and host portfolio size; subgroup metrics should be checked before applying the model uniformly.

## How to Run Locally

### 1. Setup
```bash
# clone the repo
git clone <your-repo-url>
cd airbnb-cape-town

# create and activate virtual environment
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### 2. Add data
Download the raw files into `data/raw/`:
- `data/raw/listings.csv.gz`
- `data/raw/neighbourhoods.csv`

### 3. Run the notebook
```bash
jupyter notebook
```
Open `notebooks/01_airbnb_capstone.ipynb` and run cells top-to-bottom (or Kernel Restart + Run All before submission).

### 4. Run the app
```bash
streamlit run app.py
```
The app loads `models/superhost_pipeline.joblib`. If the model file is missing, train/save it from the notebook first (Checklist L1-L3).

## Decision Log
| Decision | Options considered | Why chosen |
|---|---|---|
| Group-aware split | Random train/val/test split vs `GroupShuffleSplit` by `host_id` | Hosts have multiple listings; grouping by `host_id` prevents the same host appearing across splits, reducing leakage and giving a more honest evaluation. |
| Price parsing & cleaning | Drop all messy price rows vs parse/inspect/cap with rules | Parsed `$`/commas/blanks to numeric `price_clean`, inspected extremes row-by-row, and excluded invalid/non-positive prices from Task R while documenting impact. |
| Task C metrics | Accuracy only vs Precision/Recall/F1/ROC-AUC | Superhost labels are imbalanced; accuracy can hide poor minority-class detection. F1/precision/recall/ROC-AUC better reflect support-team tradeoffs. |
| App feature schema | Use raw `bathrooms`/`instant_bookable` vs parsed/engineered features | Final app uses 10 known-at-prediction-time features including `bathrooms_count` and `amenities_count`; `instant_bookable` was excluded because the snapshot had no usable values. |
| Deployed Task C model | Logistic Regression / Decision Tree / Random Forest / Gradient Boosting / XGBoost | XGBoost had the strongest validation F1/ROC-AUC in H8 and was saved as the app pipeline (`models/superhost_pipeline.joblib`, ~0.45 MB). The tradeoff is lower interpretability than logistic regression, so keep coefficient/importance explanations and threshold review in the notebook/app notes. |

## What I Would Do Next
- **Improve model selection/deployment:** Reconcile the validation-best model (XGBoost in H8) with the deployed app model by retraining/saving the chosen best-fit pipeline, checking calibration, and validating file size/latency.
- **Better features and validation:** Add temporal validation/newer snapshots, richer location/features, and careful text/image features (NLP/photos) with leakage controls.
- **Fairness and monitoring:** Audit subgroup performance (room type, neighbourhood, host portfolio size), add monitoring for drift, and document responsible-use limits for decision support.

## Repository Notes
- `app.py` serves **Task C only** (Superhost probability). Task R remains in the notebook.
- Do not commit `venv/` or raw data. The saved model `models/superhost_pipeline.joblib` must be included for deployment.
- Data credit: Inside Airbnb, CC BY 4.0.
