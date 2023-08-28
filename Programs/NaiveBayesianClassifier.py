#Naive Bayesian Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('Heart.csv')
df.head()
df.info()
print("Number of records in each label are")
df['target'].value_counts()
df_corr = df.corr()
plt.figure(figsize=(15,11))
sns.heatmap(df_corr, annot = True)
plt.show()
X = df.drop('target',axis=1)
y=df['target']
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42, stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_train_predict_nb = nb_clf.predict(X_train)
y_test_predict_nb = nb_clf.predict(X_test)
print('Accuracy on the training set: {:.2f}'.format(nb_clf.score(X_train, y_train)))
print('Accuracy on the test set: {:.2f}'.format(nb_clf.score(X_test, y_test)))
print(confusion_matrix(y_test, y_test_predict_nb))
print(classification_report(y_test, y_test_predict_nb))
confusion_matrix = confusion_matrix(y_test, y_test_predict_nb)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
