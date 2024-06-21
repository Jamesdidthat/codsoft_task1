# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Preprocess the data
# Let's keep it simple and use only numeric columns for now
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# Define the features and the target
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Let's say we have the following feature values for a passenger
passenger_features = {
    'Pclass': [3],
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25]
}

# Create a DataFrame with these features
passenger_df = pd.DataFrame(passenger_features)

# Use the model to make a prediction
prediction = model.predict(passenger_df)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, predictions))

# The prediction will be an array containing either 0 (did not survive) or 1 (survived)
print('Survived:', prediction[0] == 1)

