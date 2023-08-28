#Decision Tree ID3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/content/drive/MyDrive/DATASETS/Covid.csv')
df.head()
df.info()
df.columns
df=df.drop('ID',axis =1)
dic = {'YES':1, 'NO':0}
for col in df.columns:

    df[col] = df[col].map(dic)

df.head()
X = df.iloc[:,:3]
X.head()
y = df['Infected']
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42, stratify = y)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
from sklearn.tree import DecisionTreeClassifier                      #for checking testing results
from sklearn.metrics import classification_report, confusion_matrix  #for visualizing tree
from sklearn.tree import plot_tree
dtree=DecisionTreeClassifier(criterion = "entropy")
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))
cf_matrix = confusion_matrix(y_test,y_pred)
print(cf_matrix)
dec_tree = plot_tree(decision_tree=dtree, feature_names = X.columns,
                     class_names =["Infected", "Not Infected"] , filled = True , precision = 4, rounded = True)
