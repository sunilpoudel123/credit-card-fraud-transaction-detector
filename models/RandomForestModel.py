import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fraud_train = pd.read_csv("fraudTrain.csv")

fraud_train.drop("Unnamed: 0", axis=1, inplace=True)
fraud_train.drop("trans_date_trans_time", axis=1, inplace=True)
fraud_train.drop("cc_num", axis=1, inplace=True)
fraud_train.drop("merchant", axis=1, inplace=True)
# fraud_train.drop("category", axis=1, inplace=True)
# fraud_train.drop("amt", axis=1, inplace=True)
fraud_train.drop("first", axis=1, inplace=True)
fraud_train.drop("last", axis=1, inplace=True)
fraud_train.drop("gender", axis=1, inplace=True)
fraud_train.drop("street", axis=1, inplace=True)
fraud_train.drop("city", axis=1, inplace=True)
fraud_train.drop("state", axis=1, inplace=True)
fraud_train.drop("zip", axis=1, inplace=True)
fraud_train.drop("lat", axis=1, inplace=True)
fraud_train.drop("long", axis=1, inplace=True)
fraud_train.drop("city_pop", axis=1, inplace=True)
fraud_train.drop("job", axis=1, inplace=True)
fraud_train.drop("dob", axis=1, inplace=True)
fraud_train.drop("trans_num", axis=1, inplace=True)
fraud_train.drop("unix_time", axis=1, inplace=True)
# fraud_train.drop("merch_lat", axis=1, inplace=True)
# fraud_train.drop("merch_long", axis=1, inplace=True)
# fraud_train.drop("is_fraud", axis=1, inplace=True)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Defining preprocessing for categorical columns (impute missing values then apply one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['amt', 'merch_lat', 'merch_long']),
        ('cat', categorical_transformer, ['category'])
    ]
)

X_train_preprocessed = preprocessor.fit_transform(fraud_train)
y = fraud_train["is_fraud"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_preprocessed, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Extract a single tree from the forest
tree_to_plot = rf.estimators_[0]  # Select the first tree (you can choose any)

# Set the size of the plot
plt.figure(figsize=(20, 10))

# Plot the selected tree
plot_tree(tree_to_plot,
          filled=True,
          feature_names=X.columns,
          class_names=True,
          rounded=True,
          fontsize=12,
          max_depth=3,
          precision=2)

# Display the plot
plt.show()