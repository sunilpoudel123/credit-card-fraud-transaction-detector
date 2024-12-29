import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Load the dataset
data = pd.read_csv('fraudTest.csv')

data.drop(['trans_date_trans_time','dob'],axis=1)


le = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))


# Example: Assume 'target' is the column you want to predict
X = data.drop('is_fraud', axis=1)  # Features
y = data['is_fraud']               # Target variable

# Handle categorical variables if needed
# X = pd.get_dummies(X)  # This is just one approach

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)

# Set the size of the plot
plt.figure(figsize=(20, 10))

# Example with customization
plot_tree(clf, filled=True, feature_names=X.columns, class_names=True, rounded=True, fontsize=12, max_depth=3, precision=2)


# Display the plot
plt.show()
