
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


df = pd.read_csv('mental_heath_unbanlanced.csv')
my_list  = df['text'].tolist()
disorder = df['status'].tolist()


vectorizer = TfidfVectorizer(stop_words="english") 
X = vectorizer.fit_transform(my_list)


labels = pd.get_dummies(disorder).to_numpy()

xgb = XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,use_label_encoder=False,eval_metric="logloss")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=40
)

xgb.fit(X_train, labels)


# model training

# prédictions

# model evaluation
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)