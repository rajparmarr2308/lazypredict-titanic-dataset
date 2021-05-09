
import pyforest
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#data cleaning
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 2)
train.drop(columns=['Name','Ticket','Cabin', 'PassengerId', 'Parch', 'Embarked'], inplace=True)

train.dropna(inplace=True)


X = train.drop(["Survived"], axis=1)
y = train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

#let's check

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Evaluation Metrics – Random Forest:')
print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))
print('F1 Score: ' + str(metrics.f1_score(y_test, y_pred, average='macro')))


rf = LogisticRegression()
rf.fit(X_train, y_train)
y_pred_lr = rf.predict(X_test)
print('Evaluation Metrics – Logistic Regression:')
print('Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred_lr)))
print('F1 Score: ' + str(metrics.f1_score(y_test, y_pred_lr, average='macro')))