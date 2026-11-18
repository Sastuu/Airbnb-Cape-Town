# Capstone Brief: Airbnb Listings in Cape Town

**For:** Precious
**From:** Kemi
**Class:** CAMLDS Machine Learning, May 2026 cohort
**Check-in:** Tuesday 17 November 2026 at 09:00, in your normal class hour
**Presentation:** Wednesday 18 November 2026, only if the whole checklist is done (see section 1)

---

## 1. What this project is, and the deadline

You will build one machine learning project from a raw file to a live web app. You will use one real dataset that I have chosen for you. You will work on your own, in your own local environment, and you will be able to explain every step you took.

This project replaces your earlier submission. The earlier notebooks ran, but you could not explain them, and they had no real exploratory data analysis. This time the standard is different. A notebook that runs is not enough. A notebook that you understand and can defend is the goal.

**The deadline rule.**
- You present on **Wednesday 18 November** only if every item in `capstone-checklist.md` is done.
- If any item is not done by then, your presentation is moved to a later date. Nothing else happens. You finish the work and present when it is complete.
- There is one check-in with Kemi on **Tuesday 17 November at 09:00**, in your normal class hour. Kemi will confirm that sections A to G of the checklist are done, and that you have a first logistic regression and a first linear regression running. This tells both of us early whether Wednesday is realistic.
- There are two days of working time. The work is large. Start today, plan your time, and keep your notebook simple. Smaller, clear code that you understand is better than long code you cannot explain.

## 2. The problem

The Cape Town snapshot has 27,760 Airbnb listings. Hosts, property managers and the platform all make decisions about them every day. This project asks two questions about the same listings.

**Question 1, classification (the main task).** Which hosts are Superhosts? A Superhost is a host that Airbnb has given its special badge. A host support team wants to find hosts who look like Superhosts, or who are close to being one, so that it can coach the others. You will build a model that predicts `host_is_superhost` from the listing and host data.

**Question 2, regression.** What nightly price should a listing charge? A host or a property manager has to set a price for every listing. You will build a model that predicts `price`, in South African rand.

You will do both tasks on one dataset, with one round of cleaning and one round of exploratory analysis. After that, you will build models for each task, then do one clustering piece. Only the Superhost model is saved and put in a web app. The price model stays in the notebook.

## 3. The dataset

**Name.** Inside Airbnb, Cape Town listings, scraped on 23 September 2026.

**What it is.** A public snapshot of Airbnb listings in Cape Town, collected by the Inside Airbnb project. Each row is one listing. The table has 27,760 rows and 90 columns. It describes the property, the host, the price, the availability, and the guest reviews. It is real, current data, and it is not a practice dataset. It has the problems that real scraped data has, and finding and handling those problems is a large part of the work.

**Download page.** https://insideairbnb.com/get-the-data/ (find Cape Town, South Africa)

**Direct downloads.**
- Listings (the main file, about 16 MB): https://data.insideairbnb.com/south-africa/wc/cape-town/2026-09-23/data/listings.csv.gz
- Neighbourhood names (small file for the merge task): https://data.insideairbnb.com/south-africa/wc/cape-town/2026-09-23/visualisations/neighbourhoods.csv

**Licence.** Creative Commons Attribution 4.0 (CC BY 4.0). You may use the data freely, but you must credit Inside Airbnb in your README and in the app.

**Before you write any code.**
1. Read the Inside Airbnb "data assumptions" and "about" pages on their website.
2. Open the column list and write down what you think each column means.
3. Find out how Airbnb decides who becomes a Superhost. Look at Airbnb's own help page and write the rules in your leakage review. You need these rules to answer one of the hard questions in this project.

**Download the files yourself.** Save them in `data/raw` and never edit them. Pandas can read a `.csv.gz` file directly.

**The two targets.**
- Classification target: `host_is_superhost`. The values are `t` and `f`. Turn it into 1 and 0.
- Regression target: `price`. It is stored as text. You must decide how to turn it into a number, and what to do with rows that have no price.

## 4. The rules

These rules are not negotiable. They are the reason this project exists.

1. **You must be able to explain every cell out loud.** On Tuesday and on Wednesday I will pick cells at random and ask you what the code does, why you wrote it that way, and what would change if you altered it. I will also ask you to edit a cell live. If you cannot explain a cell, it does not count as done.
2. **AI may help you learn. AI must not write your notebook.** You may ask an AI tool to explain an error message, a function or a concept in simple words. You may not paste in its code and move on. If an AI tool gives you a line of code, close the chat, type the line yourself from memory, and be ready to say what each part does. Use the class notebooks, the scikit-learn documentation and your own notes first.
3. **Write your own words.** Every finding, every decision and every conclusion is written by you, in your own sentences, in the markdown cells. Short and plain is better than long and polished.
4. **Show your work over time.** Keep the project in a Git repository and commit at least at the end of each work session. A history of many small commits shows how the project grew. One large commit at the end does not.
5. **Set up your own environment.** Follow section 6. Nobody will set it up for you.
6. **Use what was taught.** The checklist maps each item to a class session. Every item must be present.
7. **Do not copy a solution.** Do not search for other people's notebooks on this dataset. If you read one, you will not be able to explain the choices in it, and I will be able to tell.
8. **Ask me early.** If you are stuck for more than 45 minutes, send me a message that says what you tried and what you saw. Being stuck is normal. Hiding it is the problem.

## 5. What you must deliver

1. A Jupyter notebook that runs from top to bottom with no manual edits, with your written findings in markdown cells.
2. One saved pipeline file (`.joblib`) for the Superhost classifier. It holds the preprocessing and the model.
3. A Streamlit app (`app.py`) that loads the Superhost pipeline and shows a Superhost probability for the listing a user describes. The app does not predict price.
4. A public GitHub repository with a README.
5. A live URL for the deployed app.
6. A presentation of about 10 minutes on Wednesday, followed by questions.

The checklist breaks these down into small items, and each item says what "done" looks like.

## 6. Set up your local environment

You will work on your own computer, not in Colab. This follows the same steps as the class setup guide (`Python-Environment-Setup-DS-ML.docx`). If you already finished that guide, you can reuse your Python install, but create a new virtual environment for this project.

**Step 1. Check Python.** Open the terminal (PowerShell on Windows, Terminal on Mac).

On Windows:
```
python --version
```
On Mac:
```
python3 --version
```
You should see a version such as 3.12. If you do not, go back to Step 1 of the setup guide.

**Step 2. Create the project folder and move into it.**
```
mkdir airbnb-cape-town
cd airbnb-cape-town
```

**Step 3. Create the virtual environment.**

On Windows:
```
python -m venv venv
```
On Mac:
```
python3 -m venv venv
```

**Step 4. Activate it.** You must do this every time you open a new terminal for this project.

On Windows:
```
.\venv\Scripts\Activate.ps1
```
On Mac:
```
source venv/bin/activate
```
You should see `(venv)` at the start of the line.

**Step 5. Update pip.**

On Windows:
```
python -m pip install --upgrade pip
```
On Mac:
```
python3 -m pip install --upgrade pip
```

**Step 6. Install the libraries.** This is the class list plus the two extra libraries you need for saving and deploying.
```
pip install numpy pandas matplotlib seaborn scikit-learn jupyter streamlit joblib
```

**Step 7. Test it.** Start Jupyter with `jupyter notebook`, create a new Python 3 notebook, and run this in a cell.
```python
import numpy, pandas, sklearn, streamlit, joblib
print("Everything works")
```

**Step 8. Start Git.**
```
git init
```
Create a file called `.gitignore` and put these lines in it, one per line, so that your virtual environment and the raw data are never uploaded.
```
venv/
data/
.ipynb_checkpoints/
__pycache__/
```
The README must tell the reader where to download the data, so the data does not need to be in the repository.

**Step 9. Write down your versions.** A saved model file only works with the same scikit-learn version that created it. Run `pip list` and copy the version numbers of `scikit-learn`, `pandas`, `numpy`, `joblib` and `streamlit` into a file called `requirements.txt` in this format.
```
streamlit==<your version>
scikit-learn==<your version>
pandas==<your version>
numpy==<your version>
joblib==<your version>
```
Write this file by hand using your own numbers. Do not use `pip freeze` for the deploy file. It lists every library, including Jupyter, and that makes the online build slow and more likely to fail. Update this file whenever you reinstall a library.

## 7. Suggested folder layout

```
airbnb-cape-town/
    README.md
    requirements.txt
    .gitignore
    app.py
    data/
        raw/
            listings.csv.gz
            neighbourhoods.csv
        processed/
    notebooks/
        01_airbnb_capstone.ipynb
    models/
        superhost_pipeline.joblib
    reports/
        figures/
```

- `data/raw` holds the original files. You never edit these. All cleaning happens in code, and the cleaned copy goes in `data/processed`.
- `models` holds the saved Superhost pipeline.
- `reports/figures` holds charts you save for the README and the presentation.

Streamlit Community Cloud reads the repository from its top folder, so keep `app.py` and `requirements.txt` in the top folder, not inside a subfolder. When your notebook is in `notebooks/`, file paths in the notebook start from that folder.

## 8. How to work

Work in this order, and do not skip ahead. Each step feeds the next.

1. **Look first.** Open the data. Write down what each column seems to mean, what you expect to be useful, and what looks wrong.
2. **Ask a specific question, then write the smallest code that answers it.** Do not write a block of code hoping it shows you something. Look at the result and decide the next question.
3. **Write your finding in a markdown cell straight after each important result.** A finding is a sentence that says what you saw and what you will do about it.
4. **Clean once and explore once, for both tasks.** Then split the data, and think carefully about which rows belong together before you split. This dataset has a trap here, and you must find it yourself.
5. **Build the simplest model first.** Add complexity only when the numbers show it helps, and only when you can say why.
6. **Keep a decision log.** In the README, record each key decision in a table with three columns: the decision, the options you considered, and why you chose what you did. Four or more entries is enough, as long as the important choices are in it.

Check what the data really contains before you trust any column name. Some columns look useful but are empty in this snapshot. Some columns hold text that looks like a number. Some columns are only known after the thing you are trying to predict has already happened. Part of your job is to find these without being told.

**One part of the split is not in the class notebooks.** You have used `train_test_split`. This dataset needs a split that keeps rows from the same group together. Search the scikit-learn documentation for how to do this. Read the page, type your own version, and be ready to explain it.

## 9. Save and deploy the app

You have already seen this in the churn project we did together. Follow the same steps on your own project.

**Step 1. Choose the inputs for the app.** A user cannot type 40 fields. Choose a small set of features, about 8 to 12, that a person could realistically enter, such as room type, number of guests, bedrooms, bathrooms, neighbourhood and minimum nights. Train the final Superhost pipeline on exactly those features, so the app never has to invent default values for columns it did not ask about. Write your reasoning in the decision log.

**Step 2. Save the whole pipeline, not only the model.** A pipeline holds the preprocessing and the model together.
```python
import joblib
joblib.dump(superhost_pipeline, "models/superhost_pipeline.joblib")
```

**Step 3. Prove that the saved file works.** In a new cell, load the file, give it a row it has seen before, and check that the prediction matches the one from the original pipeline.

**Step 4. Check the file size.** GitHub refuses single files larger than 100 MB. If the file is large, reduce the number of trees or the tree depth, or choose a smaller model. Write down what you traded away.

**Step 5. Write `app.py` in the top folder.** It must load the pipeline, show an input widget for each chosen feature, build one row with the same column names and types used in training, and show the Superhost probability with a short plain message. Add one sentence in the app that tells the user what the model cannot do, and one line that credits Inside Airbnb. You have seen a working example in the churn project. Type your own version. Do not copy it.

**Step 6. Test it locally.**
```
streamlit run app.py
```
Open the address shown in the terminal. Try at least three different inputs and confirm that the probability changes in a sensible direction.

**Step 7. Push to GitHub.** Create a public repository. It must contain `app.py`, `requirements.txt` and the file in `models/`. It should also contain your notebook and README. Do not upload the `venv` folder.

**Step 8. Deploy on Streamlit Community Cloud.** Go to share.streamlit.io and sign in with GitHub. Click "Create app", choose your repository, branch and `app.py`, then click "Deploy". Wait for the build to finish. If the build fails, read the log. It nearly always names the missing or mismatched library.

**Step 9. If Streamlit Community Cloud will not work for you,** use Hugging Face Spaces with the Streamlit option. It needs the same files.

**Step 10. Test the live URL** from your phone, then put the URL at the top of your README.

## 10. README

Your README is the front page of the project. It must contain the following, in your own words.

- A title and a one sentence description.
- The live app URL.
- The two business questions.
- The dataset, its link, how to download it, and the credit line "Data from Inside Airbnb, CC BY 4.0".
- What you found in the data, in 5 short points.
- Your final model for each task, your chosen metrics and why, and your final scores on the test set. Say which model is in the app.
- Where the models fail.
- How to run the project locally, in the exact commands.
- Your decision log, which records each key decision.
- What you would do next with more time.

## 11. The two days

This is a plan, not a rule. Adjust it, but be honest with yourself about where you are.

| When | Aim |
|---|---|
| Monday 16 Nov | Environment, data download, framing, loading, cleaning, EDA, leakage review, split, baselines (checklist sections A to G) |
| Tuesday 17 Nov, 09:00 | Check-in with Kemi. First logistic regression and first linear regression must be running. Then the rest of the models, the workflow, evaluation and error analysis (sections H to J) |
| Wednesday 18 Nov | The k-means and PCA piece, saving, the app, deployment, README and the presentation (sections K to O) |

If you are behind on Tuesday, tell Kemi at the check-in. It is better to know on Tuesday than on Wednesday.

## 12. Presentation and check-in

- **Tuesday check-in, 09:00.** Kemi checks sections A to G. He picks cells at random and asks you to explain them. He asks for one live change.
- **Wednesday presentation.** About 10 minutes covering the problem, the data and its traps, the approach, the results, the limits, and the next steps. Then questions on your notebook and a live change. The presentation only happens if the whole checklist is done.

## 13. Credit

Data from Inside Airbnb (http://insideairbnb.com), Cape Town snapshot dated 23 September 2026, licensed under CC BY 4.0.
