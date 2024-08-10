import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv('spam.csv', encoding='latin-1') 


df = df[['v1', 'v2']]
df.columns = ['label', 'text']  


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test_tfidf)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=y_pred, palette='Set2')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.title('Distribution of Predictions')
plt.show()

new_emails = ["Congratulations! You've won a $1,000 gift card. Click here to claim now.", 
              "Meeting at 10 AM tomorrow. Please confirm your availability."]
new_emails_tfidf = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_tfidf)

for email, label in zip(new_emails, predictions):
    print(f'\nEmail: {email}\nPredicted label: {"Spam" if label else "Ham"}')
