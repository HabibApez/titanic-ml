# Titanic ML

Titanic dataset analysis and survival prediction using classical machine learning models.

This project runs an end-to-end workflow on the included Titanic dataset:
1. Exploratory data analysis with plots
2. Data cleaning and basic feature engineering
3. Model training and evaluation
4. Side-by-side model comparison

The implementation is in a single script and is designed for learning and experimentation.

## Project Files

- main.py: Data loading, visualization, preprocessing, model training, and evaluation
- titanic.csv: Input dataset
- LICENSE: Project license

## What the Script Does

### 1. Loads and inspects the data

The script reads titanic.csv with pandas, then prints:
- Dataset shape
- Column names
- First rows and last rows
- Survivor and non-survivor counts and percentages

### 2. Runs exploratory visualizations

It generates multiple plots with seaborn and matplotlib, including:
- Passenger counts by class (Pclass)
- Survival by class
- Passenger and survival counts by SibSp and Parch
- Histograms for Age and Fare
- Missing-value heatmaps
- Boxplot of Age by Sex

### 3. Cleans and prepares features

Preprocessing steps implemented in the script:
- Drops columns: Name, Ticket, Embarked, PassengerId, Cabin
- Fills missing Age values using median age by Sex
- Encodes Sex into a binary IsMale column using one-hot encoding (drop_first=True)
- Drops the original Sex column

Final feature set used for training:
- Pclass
- Age
- SibSp
- Parch
- Fare
- IsMale

Target variable:
- Survived

### 4. Trains and evaluates models

The script splits data into train/test sets:
- 80% train, 20% test
- random_state=10

Models trained:
- LogisticRegression
- GaussianNB
- MultinomialNB

For each model, it reports:
- Confusion matrix
- Classification report
- Accuracy score

At the end, it prints which model performed best by accuracy (or if there is a tie).

## Requirements

Install Python 3.9+ and these packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

They are also listed in requirements.txt.

## Setup

From the project root directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Notes:
- Run from the repository root so main.py can find titanic.csv.
- The script opens several plots. Close plot windows to allow execution to finish.
- The final plt.show() call blocks until the figure window is closed.

## Current Scope and Limitations

- Single-script workflow (no modular package structure)
- No command-line arguments or config file
- No saved model artifacts
- No automated tests
- Versions are not pinned in requirements.txt

## Potential Next Improvements

- Pin package versions in requirements.txt for reproducible environments
- Refactor preprocessing and modeling into reusable functions
- Add cross-validation and additional metrics (for example ROC-AUC)
- Save trained models and plots to an output directory
- Add unit tests for preprocessing logic

## Attribution

Based on the Coursera guided project: Titanic Survival Prediction Using Machine Learning
Instructor: Ryan Ahmed
https://www.coursera.org/projects/titanic-survival-prediction-using-machine-learning
