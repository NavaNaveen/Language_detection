import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Check for null values
print(data.isnull().sum())

# Display language distribution
print(data["language"].value_counts())

# Split data into features and target
x = np.array(data["Text"])
y = np.array(data["language"])

# Convert text data to numerical data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
print("Model accuracy:", model.score(X_test, y_test))

# Predict the language of a user input
user_input = input("Enter a text: ")
data = cv.transform([user_input]).toarray()
output = model.predict(data)
print("Predicted language:", output[0])