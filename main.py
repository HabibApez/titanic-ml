# Titanic Dataset Analysis and Modeling
# This code performs data analysis and modeling on the Titanic dataset, which contains information about the passengers aboard the Titanic, 
# including whether they survived or not. The code includes data visualization, data cleaning, feature engineering, and training of logistic 
# regression and Naive Bayes models to predict survival based on the features in the dataset. 
# Finally, it assesses the performance of the trained models using confusion matrices, classification reports, and accuracy scores and 
# compares the performance of the models to determine which one performed better on the test set.
# Based on: https://www.coursera.org/projects/titanic-survival-prediction-using-machine-learning
# author: Habib Apez


# Import key libraries
from click import style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   


# Set the theme for the plots
sns.set_theme(style="ticks", palette="pastel")

# Read the data using pandas dataframe
titanic_df = pd.read_csv('titanic.csv')

# Print the number of rows and columns in the dataset
print("Shape of the dataset:", titanic_df.shape)
# Print the columns in the dataset, exploring the features of the dataset
print("Columns in the dataset:", titanic_df.columns)

# Show the data head (first element of the dataset)
print(titanic_df.head())
# Show the last 7 rows of the dataset
print(titanic_df.tail(7))

# DATA VISUALIZATION

# Let's coun tthe numbers of survivors and non-survivors in the dataset
survived_df = titanic_df[ titanic_df ["Survived"] == 1]
print("Number of survivors:", survived_df.shape[0])
no_survived_df = titanic_df [titanic_df ["Survived"] == 0]
print("Number of non-survivors:", no_survived_df.shape[0]) 

print("Survivors:")
print(survived_df)

# Count the survivors and deceased passengers
print("Total:", len(titanic_df))

print("Number of passengers who survived:", len(survived_df))
print("Percentage of passengers who survived:", 1.0* len(survived_df) / len(titanic_df) * 100, "%")

print("Number of passengers who did not survive:", len(no_survived_df))
print("Percentage of passengers who did not survive:", 1.0* len(no_survived_df) / len(titanic_df) * 100, "%")

# Bar char to indicate the number of people who survived based ontheir clas
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
# Subplot 1: Count of passengers based on their class
sns.countplot(x="Pclass", data=titanic_df, ax=axes[0])
axes[0].set_title("Count of Passengers by Class")
# Subplot 2: Count of passengers based on their class who survived
sns.countplot(x="Pclass", hue="Survived", data=titanic_df, ax=axes[1])
axes[1].set_title("Count of Survivors by Class")
plt.tight_layout()
plt.show(block=False)

# Plot the number of people who survived based on their SibSp (number of siblings/spouses aboard)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
# Subplot 1: Count of passengers based on their SibSp
sns.countplot(x="SibSp", data=titanic_df, ax=axes[0])
axes[0].set_title("Count of Passengers by SibSp")
# Subplot 2: Count of passengers based on their SibSp who survived
sns.countplot(x="SibSp", hue="Survived", data=titanic_df, ax=axes[1])
axes[1].set_title("Count of Survivors by SibSp")
plt.tight_layout()
plt.show(block=False)

# Plot the number of people who survived based on their Parch (number of parents/children aboard)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
# Subplot 1: Count of passengers based on their Parch
sns.countplot(x="Parch", data=titanic_df, ax=axes[0])
axes[0].set_title("Count of Passengers by Parch")   
# Subplot 2: Count of passengers based on their Parch who survived
sns.countplot(x="Parch", hue="Survived", data=titanic_df, ax=axes[1])
axes[1].set_title("Count of Survivors by Parch")
plt.tight_layout()
plt.show(block=False)

# Age histogram to show the distribution of ages of passengers who survived and did not survive
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
titanic_df["Age"].hist(bins=100, ax=axes[0])
axes[0].set_title("Distribution of Ages of Passengers")

# Fare histogram to show the distribution of fares paid by passengers who survived and did not survive
titanic_df["Fare"].hist(bins=100, ax=axes[1])
axes[1].set_title("Distribution of Fares Paid by Passengers")
plt.tight_layout()
plt.show(block=False)

# CLEANING THE DATA AND FEATURE ENGINEERING

# Check for missing values in the dataset
print("Missing values in each column:")
print(titanic_df.isnull().sum())

# Visualize the missing values in the dataset using a heatmap
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
sns.heatmap(titanic_df.isnull(), yticklabels=False, cbar=False, cmap="Blues", ax=axes[0])
axes[0].set_title("Heatmap of missing values in the Dataset")

# Drop the columns that are not useful for analysis and modeling, such as "Name", "Ticket", "Embarked", "PassengerId" and "Cabin" 
titanic_df_cleaned = titanic_df.drop(columns=["Name", "Ticket", "Embarked", "PassengerId", "Cabin"], axis=1, inplace=False)
print("Columns in the cleaned dataset:", titanic_df_cleaned.columns)
print("Shape of the cleaned dataset:", titanic_df_cleaned.shape)
print(titanic_df_cleaned.head())

# Visualize the missing values in the cleaned dataset using a heatmap
sns.heatmap(titanic_df_cleaned.isnull(), yticklabels=False, cbar=False, cmap="Blues", ax=axes[1])
axes[1].set_title("Heatmap of missing values in the cleaned Dataset")
plt.tight_layout()
plt.show(block=False)

print(titanic_df_cleaned)


fig, axes = plt.subplots(1, 1, figsize=(15, 10))
# Subplot 1
sns.boxplot(x="Sex", y="Age", data=titanic_df_cleaned, ax=axes)
axes.grid(True)
plt.tight_layout()
plt.show(block=False)

# Fill the missing values in the "Age" column with the median age of the passengers
median_age_male = titanic_df_cleaned[titanic_df_cleaned["Sex"] == "male"]["Age"].median()
median_age_female = titanic_df_cleaned[titanic_df_cleaned["Sex"] == "female"]["Age"].median()

# Show the median ages for male and females
print("Median age for male passengers:", median_age_male)
print("Median age for female passengers:", median_age_female)

def fill_missing_ages(df):
    age = df["Age"]
    sex = df["Sex"]
    if pd.isnull(age):
        if sex == "male":
            return median_age_male
        else:
            return median_age_female
    else:
            return age

# Apply the function to fill the missing values in the "Age" column
titanic_df_cleaned["Age"] = titanic_df_cleaned.apply(fill_missing_ages, axis=1)

print("Missing values in the cleaned dataset after filling missing ages:")
print(titanic_df_cleaned.isnull().sum())
print(titanic_df_cleaned)

# Visualize the heatmap of cleaned dataset after filling the missing values
fig, axes = plt.subplots(1, 1, figsize=(15, 10))
sns.heatmap(titanic_df_cleaned.isnull(), yticklabels=False, cbar=False, cmap="Blues", ax=axes)
axes.set_title("Heatmap of missing values in the cleaned Dataset after filling missing ages")
plt.tight_layout()
plt.show(block=False)

print(titanic_df_cleaned)

# We just need one column to represent male or female, so we can drop the "Sex" column and create a new column "IsMale"
# get|dummies function and drop the first column to avoid multicollinearity
print(pd.get_dummies(titanic_df_cleaned["Sex"]))
IsMale = pd.get_dummies(titanic_df_cleaned["Sex"], drop_first=True).astype(int)
print(IsMale)
IsMale.rename(columns={"male": "IsMale"}, inplace=True) 

# Concatenate the "IsMale" column with the cleaned dataset
titanic_df_cleaned = pd.concat([titanic_df_cleaned, IsMale], axis=1)
print(titanic_df_cleaned)

# Drop the "Sex" column from the cleaned dataset
titanic_df_cleaned.drop(columns=["Sex"], inplace=True)
print(titanic_df_cleaned)   

# TRAIN LOGISTIC REGRESSION MODEL

# Split the dataset into features (X) and target variable (y)
X = titanic_df_cleaned.drop(columns=["Survived"], axis=1).values
y = titanic_df_cleaned["Survived"].values


from sklearn.model_selection import train_test_split 
# Split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing, and set a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LogisticRegression
# Create an instance of the LogisticRegression class and fit the model to the training data
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

print(classifier)
print("Logistic Regression model trained successfully.")

# ASSESS TRAINED MODEL PERFORMANCE

# Predict the values for the test set using the trained logistic regression model
y_predict_test = classifier.predict(X_test)
print("Predicted values for the test set:")
print(y_predict_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Calculate the confusion matrix, classification report, and accuracy score for the test set predictions
conf_matrix = confusion_matrix(y_test, y_predict_test)
class_report = classification_report(y_test, y_predict_test)
accuracy = accuracy_score(y_test, y_predict_test)

print("Confusion Matrix:")
print(conf_matrix)

fig, axes = plt.subplots(3, 1,figsize=(15, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix for Logistic Regression Model")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("True Values")

print("Classification Report:")
print(class_report)
print("Accuracy Score:", accuracy)  

# Train the Naive Bayes models and assess their performance using the same test set to compare it with the Logistic Regression model

from sklearn.naive_bayes import GaussianNB
# Create an instance of the GaussianNB class and fit the model to the training data
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
print(nb_classifier)
print("Naive Bayes model trained successfully.")

y_predict_test_nb = nb_classifier.predict(X_test)
print("Predicted values for the test set using Naive Bayes model, GaussianNB:")
print(y_predict_test_nb)

conf_matrix_nb = confusion_matrix(y_test, y_predict_test_nb)
class_report_nb = classification_report(y_test, y_predict_test_nb)
accuracy_nb = accuracy_score(y_test, y_predict_test_nb)

sns.heatmap(conf_matrix_nb, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title("Confusion Matrix for Naive Bayes Model, GaussianNB")
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("True Values")

print("Classification Report for Naive Bayes Model, GaussianNB:")
print(class_report_nb)
print("Accuracy Score for Naive Bayes Model, GaussianNB:", accuracy_nb)  


from sklearn.naive_bayes import MultinomialNB
# Create an instance of the MultinomialNB class and fit the model to the training data
mnb_classifier = MultinomialNB()
mnb_classifier.fit(X_train, y_train)
print(mnb_classifier)
print("Multinomial Naive Bayes model trained successfully.")

print("Predicted values for the test set using Naive Bayes model, MultinomialNB:")
y_predict_test_mnb = mnb_classifier.predict(X_test)
print(y_predict_test_mnb)

confusion_matrix_mnb = confusion_matrix(y_test, y_predict_test_mnb)
class_report_mnb = classification_report(y_test, y_predict_test_mnb)
accuracy_mnb = accuracy_score(y_test, y_predict_test_mnb)

sns.heatmap(confusion_matrix_mnb, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_xlabel("Predicted Values")
axes[2].set_ylabel("True Values")

plt.tight_layout()
plt.show()

print("Classification Report for Naive Bayes Model, MultinomialNB:")
print(class_report_mnb)
print("Accuracy Score for Naive Bayes Model, MultinomialNB:", accuracy_mnb)

# Which model performed better, Logistic Regression or Naive Bayes GaussianNB or Naive Bayes MultinomialNB? 
# Based on the accuracy scores and classification reports, we can compare the performance of the three models 
# and determine which one performed better on the test set.

# Determine which model performed better based on the accuracy scores
if accuracy > accuracy_nb and accuracy > accuracy_mnb:
    print("The Logistic Regression model performed better than both Naive Bayes models.")
elif accuracy_nb > accuracy and accuracy_nb > accuracy_mnb:
    print("The Naive Bayes model, GaussianNB, performed better than both the Logistic Regression model and the MultinomialNB model.")
elif accuracy_mnb > accuracy and accuracy_mnb > accuracy_nb:
    print("The Naive Bayes model, MultinomialNB, performed better than both the Logistic Regression model and the GaussianNB model.")
else:
    print("There is a tie in performance between the models based on accuracy scores.")
