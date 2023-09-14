import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv(r"D:\Prakash\FinalBalancedDataset.csv")
model = LabelEncoder()
df['diagnosis'] = model.fit_transform(df['diagnosis'])
x = df.drop(['diagnosis'],axis = 1)
y = df['diagnosis']
df.['diagnosis'] value_counts(normalize =True)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_train_tfidf = tfidf_vectorizer.fit_transform(y_train
y_test_tfidf = tfidf_vectorizer.transform(y_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
report = classification_report(y_test, y_pred)
print(report)
