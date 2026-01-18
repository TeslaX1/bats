
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pandas as pd
df = pd.read_csv('mental_heath_unbanlanced.csv')
my_list  = df['text'].tolist()
disorder = df['status'].tolist()

texts = [
"machine learning is fascinating",
"this topic is very boring",
"i enjoy studying data science",
"this subject is terrible"
]
labels = [1, 0, 1, 0] # 1 = Positive, 0 = Negative
vectorizer = TfidfVectorizer(stop_words="english") 
X = vectorizer.fit_transform(texts)
# Y = vectorizer.fit_transform(my_list)
# Z = vectorizer.fit_transform(disorder)
xgb = XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,use_label_encoder=False,eval_metric="logloss")
xgb.fit(X, labels)
# xgb.fit(Y,Z)

print(vectorizer.fit_transform(['Anxiety','Depression','Suicidal','Normal']))
print(xgb.predict('I am very happy'))
print(xgb.predict("What is machine learning"))


