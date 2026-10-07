# From Words to Labels

Sentiment classification of Amazon product reviews with classical machine learning. The project takes about 3.6 million reviews labelled positive or negative, cleans and lemmatizes the text, turns it into TF-IDF features plus three simple numeric features, and compares five classifiers and a voting ensemble on a held-out test set of about 400,000 reviews. The best model, a linear support vector classifier (Linear SVC), reached 89.78% test accuracy. It is served by a small Flask app with an HTML/JavaScript page where you can type a review and get a positive or negative label.

## Dataset

[Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) (Kaggle), in fastText format: each line starts with `__label__1` (1 or 2 stars, negative) or `__label__2` (4 or 5 stars, positive), followed by the review title and text. 3-star reviews are not included. The raw data files are not stored in this repository (see "How to run").

| Split | Reviews in file | After keeping English only |
|---|---|---|
| Train (`train.ft.txt`) | 3,600,000 (1,800,000 per class) | 3,586,509 (1,794,975 negative, 1,791,534 positive) |
| Test (`test.ft.txt`) | 400,000 (200,000 per class) | 398,473 (199,417 negative, 199,056 positive) |

## Pipeline

1. Parse each line into the review text and its label (1 = negative, 2 = positive).
2. Detect the language with `langid` and keep English reviews only.
3. Expand contractions (`contractions`), lowercase, and remove punctuation except `!` and `?`.
4. Convert numbers to words (`inflect`) and flag reviews where this failed.
5. Lemmatize with spaCy (`en_core_web_sm`) and remove NLTK English stopwords, keeping the negations "not", "no", "nor" and "n't".
6. Build features: TF-IDF on unigrams (`max_features=10000`, `min_df=6`, `max_df=0.9`) plus three numeric features: a flag for failed number conversion, the word count, and the number of `!` and `?` (both counted on the cleaned text). That gives 10,003 features per review. Two training reviews were empty after cleaning and were dropped, leaving 3,586,507 for training.
7. Apply the same steps to the test set, using the TF-IDF vectorizer fitted on the training set.

The exploration notebooks also plot the 20 most common words, a word cloud for each class, and the 10 features most correlated with the label. The top three by absolute correlation are "not" (0.337), "great" (0.269) and "love" (0.201), which supports keeping negations in step 5.

## Models compared

All models were trained with scikit-learn in `buildingModel.ipynb`:

- Logistic Regression (L2 penalty, `saga` solver, `max_iter=3000`)
- Linear SVC (L2 penalty, squared hinge loss, `C=1.0`)
- Random Forest (200 trees, `max_depth=30`) on 200 TruncatedSVD components
- Ridge Classifier (`alpha=1.0`, `sag` solver)
- Multinomial Naive Bayes (best of `alpha` in {0, 0.5, 1.0} and `fit_prior` in {True, False}: `alpha=1.0`, `fit_prior=True`)
- Hard-voting ensemble of the five models above (majority vote)

## Results

Test set: 398,473 English reviews. Precision, recall and F1 are the macro averages printed by scikit-learn's `classification_report` (two decimals).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Linear SVC (best)** | **89.78%** | **0.90** | **0.90** | **0.90** |
| Voting ensemble | 89.29% | 0.89 | 0.89 | 0.89 |
| Ridge Classifier | 89.11% | 0.89 | 0.89 | 0.89 |
| Logistic Regression | 88.53% | 0.89 | 0.89 | 0.89 |
| Multinomial Naive Bayes | 83.93% | 0.84 | 0.84 | 0.84 |
| Random Forest (SVD features) | 82.69% | 0.83 | 0.83 | 0.83 |

Linear SVC confusion matrix: 178,703 negative and 179,056 positive reviews were classified correctly; 20,714 negative reviews were labelled positive and 20,000 positive reviews were labelled negative.

Notes: Logistic Regression stopped at its iteration limit before converging. There is no separate validation set; the Naive Bayes settings were picked by test accuracy, and all numbers come from the single test split.

## How to run

### Requirements

The notebooks were run with Python 3.11.9, and the saved models were created with scikit-learn 1.6.1. There is no requirements file; these are the packages imported by the notebooks and the app:

```
pip install pandas numpy scipy scikit-learn==1.6.1 joblib tqdm langid contractions inflect spacy nltk matplotlib seaborn wordcloud flask
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

`TEST_tfidfPlusFeatures.npz` (about 119 MB) is stored with Git LFS, so install Git LFS before cloning to get the real file.

### Run the web app

The trained Linear SVC and the fitted vectorizer are included, so the app runs without retraining:

```
cd amazonreviews/versions
python app.py
```

Open http://127.0.0.1:5000/, type a review and click Analyze. The page sends `{"review": "..."}` to `POST /predict` and shows the returned label; input that is not detected as English is rejected. Start the app from `amazonreviews/versions`, because it loads the model files by relative path. It runs Flask's development server, which is meant for local use only.

### Reproduce the pipeline

Download the dataset from Kaggle (decompress the files if they come as `.bz2`), then put `train.ft.txt` in `amazonreviews/versions/train.ft.txt/` and `test.ft.txt` in `amazonreviews/versions/test.ft.txt/`, replacing the placeholder text files. Run the notebooks in this order:

1. `train.ft.txt/1) creatingDataFrame.ipynb`: parse the raw file into `trainDataFrame.csv`
2. `train.ft.txt/2) preprocessLanguageFilter.ipynb`: detect the language of each review
3. `train.ft.txt/3) textPreprocessingAndFeatureEngineering.ipynb`: keep English, clean, lemmatize, add features, plot word counts
4. `train.ft.txt/4) vectorization.ipynb`: fit TF-IDF and save the vectorizer and the feature matrix
5. `train.ft.txt/5) visualizeTop10.ipynb` (optional): top 10 features correlated with the label
6. `test.ft.txt/samePipelineOnTesting.ipynb`: the same steps for the test set
7. `buildingModel.ipynb`: train, evaluate and save the models

Things to know before running:

- Notebooks 1 to 5 read and write files in their own folder. `buildingModel.ipynb`, `app.py` and the last code cell of `samePipelineOnTesting.ipynb` use paths relative to `amazonreviews/versions/`, some written with Windows backslashes, so adjust the working directory or the paths.
- `buildingModel.ipynb` expects the test matrix at `test.ft.txt/TEST_tfidfPlusFeatures.npz`, but the repository keeps it in `amazonreviews/versions/`. Move it or edit the path.
- The training feature matrix is not in the repository, so retraining needs steps 1 to 4 first. The Random Forest model (about 9 GB) was not committed either, so the voting cell needs it retrained.
- Expect long run times: in the saved run, lemmatizing the training set took about 3 hours 42 minutes and training Logistic Regression took about 3.9 hours.

## Repository layout

```
amazonreviews/
  7.complete                    empty marker file
  versions/
    app.py                      Flask app: serves the page and the /predict endpoint
    templates/index.html        HTML/JavaScript page
    HowToRun.txt                short instructions for the app
    buildingModel.ipynb         model training and evaluation
    *.joblib                    saved models and the TruncatedSVD transformer
    TEST_tfidfPlusFeatures.npz  test-set feature matrix (Git LFS)
    train.ft.txt/               notebooks 1-5, fitted TF-IDF vectorizer, feature names
    test.ft.txt/                test-set notebook and feature names
```

Saved models: `linear_svc_l2_model.joblib` (used by the app), `logistic_regression_l2_model.joblib`, `best_ridge_classifier_model.joblib`, `best_multinomial_nb_model.joblib`, and `svd_transformer.joblib` (TruncatedSVD for the Random Forest).

## Author

Fares Al-Fares ([github.com/F2-116](https://github.com/F2-116))
