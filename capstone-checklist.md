# Capstone Checklist: Airbnb Cape Town

**Student:** Precious
**Reviewer:** Kemi

**The rule.** Precious presents on Wednesday 18 November only if every box below is ticked. If any box is not ticked, the presentation is moved.

**How to use this file.** Precious ticks a box only when the item meets the "Done looks like" line. Kemi makes the final tick. An item is not done if Precious cannot explain it out loud.

**Two tasks, one dataset.** The app serves Task C only. Task R lives in the notebook.
- **Task C (classification).** Predict `host_is_superhost`.
- **Task R (regression).** Predict `price`.

Cleaning, EDA and the split are done once and shared by both tasks. Sections H, I and J mark which task each item is for.

**Check-in on Tuesday 17 November at 09:00, in his normal class hour.** Kemi confirms sections A to G, plus H1 and H5 (the first logistic regression and the first linear regression). The session column shows where each skill was taught.

---

## A. Environment and repository (setup guide)

- [ ] **A1. Virtual environment created and working**
  Done looks like: a `venv` folder exists, `(venv)` shows in the terminal, and the import test cell prints "Everything works". Precious ran it in front of Kemi.
- [ ] **A2. Git repository with a `.gitignore`**
  Done looks like: `git log` shows several commits across both days. `venv/` and `data/` are not in the repository.
- [ ] **A3. Folder layout in place**
  Done looks like: the raw files sit in `data/raw` and have not been edited. Cleaned data is written by code into `data/processed`.
- [ ] **A4. Hand written `requirements.txt` with pinned versions**
  Done looks like: it lists five libraries with version numbers that match `pip list` on Precious's machine.

## B. Problem framing (Session 10)

- [ ] **B1. Both business questions in his own words**
  Done looks like: a markdown cell at the top says who uses each prediction and what decision it supports. It says why Task C is classification and why Task R is regression.
- [ ] **B2. Targets defined**
  Done looks like: `host_is_superhost` is turned into 1 and 0 with the class counts printed. `price` is turned into a number. A written note says what happens to rows with no price.
- [ ] **B3. Metrics chosen before modelling, with reasons**
  Done looks like: for Task C, at least two metrics are named and a sentence says why accuracy alone is not enough at this class balance. For Task R, RMSE and MAE are named and a sentence says how they differ. Each choice is linked to the business question.

## C. Data loading and inspection (Sessions 1 and 2)

- [ ] **C1. Data loaded from the raw file**
  Done looks like: `listings.csv.gz` is read with pandas, the shape is printed (27,760 rows and 90 columns), and he can say how pandas handled the `.gz` file.
- [ ] **C2. Inspection with `info`, `describe` and `value_counts`**
  Done looks like: every column has been looked at. A markdown cell lists, in his words, which columns are numbers, categories, text, or not what they seem.
- [ ] **C3. Dtypes fixed**
  Done looks like: `price` (text with a currency sign and commas) is cast to a number. `bathrooms_text` is turned into a bathroom count. Each cast is shown before and after, and he can explain why pandas read the column as text.
- [ ] **C4. Missing values audited**
  Done looks like: a table of missing counts and percentages for all columns. Columns that are completely empty in this snapshot are identified. Each column with missing data has a written decision (drop it, fill it, or keep the gap as information) and a reason.
- [ ] **C5. Duplicates and repeats checked**
  Done looks like: he checked for duplicate rows, counted how many hosts have more than one listing, and wrote down what that means for the split and for any host-level column.
- [ ] **C6. Merge with the neighbourhood file**
  Done looks like: `neighbourhoods.csv` is merged onto the listings with `merge`, with the join type named and a check that no rows were lost or duplicated. He can explain what would change with a different join type.

## D. Cleaning (Sessions 1 and 7)

- [ ] **D1. Cleaning done in code, in order, in one place**
  Done looks like: one clear block of cells goes from raw to processed. Re-running it from the top gives the same file.
- [ ] **D2. Useless columns removed with a reason**
  Done looks like: identifiers, web links, empty columns and free text that will not be used are dropped, with one line of reason for each group.
- [ ] **D3. Rows removed only with a reason**
  Done looks like: any removed rows are counted. Extreme prices and extreme minimum stays are looked at row by row before a decision is made, and he can say why those rows should or should not be in a model.
- [ ] **D4. Skew handled for price**
  Done looks like: the raw price distribution and a transformed version (for example the log) are plotted side by side. He can say why the transformed target is used or not used, and how a prediction is turned back into rand.
- [ ] **D5. One engineered feature with a before and after**
  Done looks like: at least one new feature built from existing columns (for example the number of amenities from the amenities list, or the bathroom count). A written reason says why it should help, and its effect is measured later on validation data.

## E. EDA with written findings (Session 2)

- [ ] **E1. Both targets plotted and described**
  Done looks like: a chart of the Superhost split and a histogram of price, each with a sentence on balance or shape.
- [ ] **E2. At least four distribution plots**
  Done looks like: histograms or boxplots of key numbers (for example price, accommodates, minimum nights and review scores). Each is followed by a sentence on shape, skew and outliers.
- [ ] **E3. At least four plots of feature against target**
  Done looks like: Superhost rate by group (for example by review rating band or by room type) and price by group (for example by room type or neighbourhood), each with a written reading.
- [ ] **E4. A groupby summary**
  Done looks like: `groupby` with `agg` is used at least once to build a table (for example median price and Superhost share by neighbourhood), with a sentence on what stands out.
- [ ] **E5. A correlation check for the numeric columns**
  Done looks like: a correlation table or heatmap and one sentence on any pair that moves together. He names any column that looks too related to a target and says what he will do about it.
- [ ] **E6. The five point EDA narrative**
  Done looks like: five numbered findings in his own words covering what the data contains, what stands out, what needs cleaning, what looks risky for modelling, and what he plans to try. Each finding points to the plot or number it came from.

## F. Split and leakage check (Session 3)

- [ ] **F1. Leakage review written down before any model**
  Done looks like: a markdown list goes through every column that could be used and says, for each task, whether it would be known at prediction time and whether it gives away the target. Airbnb's Superhost rules are quoted or summarised. Columns that leak are removed from the relevant task with a reason.
- [ ] **F2. Group aware split**
  Done looks like: the split keeps all listings from the same host in the same set. He can explain what would go wrong if one host's listings were on both sides, and he has shown that it matters by comparing it with a plain random split on one model.
- [ ] **F3. Train, validation and test sets with a fixed seed**
  Done looks like: three sets with sizes printed. For Task C, the Superhost rate is printed for each set. The test set is not touched again until section J.
- [ ] **F4. Preprocessing fitted on training data only**
  Done looks like: scalers, encoders and imputers are inside a pipeline, or fitted on the training set alone. Any feature built from group statistics (for example the average price of a neighbourhood) is built from training rows only. He can say what leakage would happen otherwise.

## G. Baselines (Session 3)

- [ ] **G1. Majority class baseline for Task C**
  Done looks like: a `DummyClassifier` or equivalent result on validation data with accuracy, precision, recall and F1. He can say why the accuracy looks reasonable while the recall is zero.
- [ ] **G2. Mean (or median) predictor baseline for Task R**
  Done looks like: a predictor that always gives the training mean or median, scored with RMSE and MAE on validation data. This is the number every price model must beat.
- [ ] **G3. A bias and variance sentence**
  Done looks like: a written plain sentence says how a baseline, a simple model and an overfit model sit on the bias and variance idea. He backs it with a number from his own results later in section H.

## H. Linear and tree models (Sessions 4 and 5)

- [ ] **H1. Logistic regression for Task C**
  Done looks like: fitted inside a pipeline with scaling and encoding and scored on validation data against the baseline.
- [ ] **H2. Logistic regression coefficients and `C`**
  Done looks like: the ten largest coefficients are listed and he explains in words what two of them mean. At least three values of `C` are compared with a reading of what changed. He can say what L1 and L2 penalize.
- [ ] **H3. Decision tree for Task C, depth curve and a plot of the tree**
  Done looks like: training and validation scores are plotted across several depths, with the point where overfitting starts named. One shallow tree is drawn with `plot_tree` and he reads the first split out loud.
- [ ] **H4. Random forest and gradient boosting for Task C**
  Done looks like: both are fitted and added to one results table with the logistic regression and the baseline. He can say how bagging differs from boosting in two sentences.
- [ ] **H5. Linear regression for Task R, with coefficients**
  Done looks like: fitted inside a pipeline, scored on validation data against the mean baseline. Two coefficients are explained in words, and he says what a coefficient means when the target is log price (if he used it).
- [ ] **H6. Ridge and Lasso for Task R**
  Done looks like: both are fitted on the same data, and the coefficients are compared with plain linear regression. He shows which coefficients shrink or reach zero and can say why.
- [ ] **H7. Feature importance for Task C**
  Done looks like: importances from the best tree model for Task C are plotted. A sentence compares them with the logistic regression coefficients, and a sentence says why importance is not the same as cause.

## I. Workflow: Pipeline, CV and tuning (Session 7)

- [ ] **I1. One `Pipeline` with a `ColumnTransformer` per task**
  Done looks like: numeric and categorical columns get different preprocessing, and the model is the last step. The same object is used for CV, tuning, and saving.
- [ ] **I2. Cross validation that respects hosts**
  Done looks like: k fold results with mean and spread, using a splitter that keeps each host in one fold (`GroupKFold`). He can explain why a plain k fold would give a score that is too good.
- [ ] **I3. Hyperparameter search**
  Done looks like: `RandomizedSearchCV` or `GridSearchCV` on a stated search space, run for the best model family of each task. The best parameters and the CV score are shown, with a sentence on whether tuning helped compared with the default.
- [ ] **I4. Imbalance handling for Task C**
  Done looks like: at least one approach (class weights, or a changed decision threshold) is tried, with a before and after on precision, recall and F1.
- [ ] **I5. The engineered feature measured**
  Done looks like: the feature from D5 is compared with and without it on validation data, with a sentence on whether it helped.

## J. Evaluation and error analysis (Session 6)

- [ ] **J1. Final test scores, once per task**
  Done looks like: the test set is scored one time with each final model. Task C has accuracy, precision, recall, F1 and ROC-AUC. Task R has RMSE and MAE in rand. He can explain why scoring the test set many times would be a mistake.
- [ ] **J2. Confusion matrix read in words**
  Done looks like: the four counts are labelled for Task C. He says what a false alarm and a missed Superhost each cost the support team.
- [ ] **J3. ROC curve and precision recall curve**
  Done looks like: both curves are plotted for the final Task C model. He can say what the ROC-AUC value means (including what 0.5 means) and why the precision recall curve is useful at this class balance.
- [ ] **J4. Threshold choice tied to the business**
  Done looks like: precision and recall are shown at several thresholds, and a chosen threshold has a written reason that mentions the size of the support team.
- [ ] **J5. Error analysis by hand, for both tasks**
  Done looks like: for each task, 10 of the worst predictions are inspected row by row, grouped by what they have in common, with a written guess about why the model failed on each group.
- [ ] **J6. Performance by subgroup**
  Done looks like: scores are compared across at least two groups (for example room type or neighbourhood area), with a sentence on whether the models work equally well for all of them.
- [ ] **J7. Limits stated**
  Done looks like: a short list of what the models cannot be trusted for, for example that the data is one snapshot, covers one city, and does not include listing photos or text.

## K. Unsupervised piece (Session 8)

- [ ] **K1. K-means on listings**
  Done looks like: k-means on scaled numeric features, with the number of clusters chosen using an elbow plot and the silhouette score. The reasoning is written down and the centroids are read in plain words.
- [ ] **K2. PCA picture**
  Done looks like: a two component plot coloured by cluster, with the explained variance printed and read.
- [ ] **K3. Clusters connected to the targets**
  Done looks like: the Superhost rate and the median price per cluster, with a plain description of each cluster and a sentence on whether the clusters add anything the supervised models do not already know.

## L. Saved pipelines (Block 5 of the churn project)

- [ ] **L1. Whole Superhost pipeline saved with `joblib`**
  Done looks like: `models/superhost_pipeline.joblib` exists and holds the preprocessing and the model. The price model stays in the notebook and is not saved.
- [ ] **L2. Reload test**
  Done looks like: a fresh cell loads the file and gives the same prediction as the original for the same row. The file is under 100 MB.
- [ ] **L3. Final model trained on the app features only**
  Done looks like: the deployed pipeline uses the 8 to 12 features that the app asks for, and the decision log says why those were chosen.

## M. Streamlit app

- [ ] **M1. `app.py` written by him**
  Done looks like: Kemi asks him to add one new input widget live, and he does it without help.
- [ ] **M2. Runs locally**
  Done looks like: `streamlit run app.py` works, and at least three different inputs give a sensible Superhost probability.
- [ ] **M3. Plain message, a stated limit and the credit line**
  Done looks like: the app shows the Superhost probability with a short message in words. It has one sentence about what the model cannot do and one line crediting Inside Airbnb.
- [ ] **M4. Input handling**
  Done looks like: the row the app builds has the same column names and types as the training data, and he can explain what happens if one column is missing.

## N. Deployment

- [ ] **N1. Public GitHub repository**
  Done looks like: contains `app.py`, `requirements.txt`, the model file, the notebook and the README. It does not contain `venv` or the raw data.
- [ ] **N2. Deployed URL works**
  Done looks like: Kemi opens the link on his own phone and gets a Superhost probability.
- [ ] **N3. Deploy problem explained**
  Done looks like: he can describe at least one error from the build log or from local testing and how he fixed it. If nothing failed, he can explain why the versions in `requirements.txt` matter.

## O. README and presentation

- [ ] **O1. README complete**
  Done looks like: it has every item listed in section 10 of the brief, including the live URL, the credit line, and a decision log that records each key decision, with at least 4 entries.
- [ ] **O2. Notebook runs from top to bottom**
  Done looks like: Kemi restarts the kernel and runs all cells. There are no errors and no manual edits needed.
- [ ] **O3. Notebook reads as a story**
  Done looks like: every section has a heading, and every important result has a written finding in his own words. There are no cells without a purpose.
- [ ] **O4. 10 minute presentation**
  Done looks like: it covers the problem, the data and its traps, the approach, the results, the limits and the next steps. He does not read from the screen.
- [ ] **O5. Live questions answered**
  Done looks like: he answers Kemi's questions on random cells and one live code change, and he can say "I do not know" honestly when he does not.

---

## Coverage map against the class curriculum

| Session | What was taught | Where it is in this checklist | Not checked in this project |
|---|---|---|---|
| 1 | Load files, dtypes, missing data, `groupby` and `agg`, `merge` | C1 to C6, D1 to D3, E4 | None |
| 2 | `describe`, `info`, `value_counts`, plots, outliers, EDA narrative | C2, E1 to E6 | None |
| 3 | Train, validation, test, leakage, baselines, bias and variance | F1 to F4, G1 to G3 | None |
| 4 | Linear and logistic regression, L1 and L2, gradient descent | H1, H2, H5, H6 | Gradient descent in words |
| 5 | Decision tree, random forest, gradient boosting, overfitting at depth | H3, H4, H7 for Task C | Tree models for the price task |
| 6 | Precision, recall, F1, ROC-AUC, RMSE, MAE, confusion matrix, error analysis | B3, G1, G2, J1 to J7 | None |
| 7 | k fold CV, search, `Pipeline`, `ColumnTransformer`, feature engineering | D5, I1 to I5 | None |
| 8 | K-means, elbow and silhouette, PCA | K1 to K3 | Hierarchical clustering and dendrogram, anomaly detection |
| 9 | Recommenders | Not used. The dataset has no user and item ratings. | All of it |
| 10 | End to end project | The whole checklist, B1 | None |
| 11 | Presentation and feedback | O4, O5 | None |

The precision recall curve and threshold choice (J3, J4) go slightly beyond the class notes but build on Session 6. Saving and deploying come from the churn project, not from a numbered session.
