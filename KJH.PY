
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# dataset loading

X, y = my_file, dataset.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Logistic Regression model initialization
model = LogisticRegression(
    penalty='l2',
    C=2.0,
    solver='liblinear',
    max_iter=1000
)

# model training
model.fit(X_train, y_train)

# prédictions
y_pred = model.predict(X_test)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)