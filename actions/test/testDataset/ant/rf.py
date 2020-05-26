import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('method.csv')

print(df.head())
df['isBuggy'].value_counts()

xTrain = df.drop(['isBuggy',"path"], axis=1)
yTrain = df['isBuggy']
(xTrain, xTest , yTrain, yTest) = train_test_split(xTrain, yTrain, test_size = 0.3, random_state = 42)

parameters = {
    'n_estimators' :[10,50,100,1000],#作成する決定木の数
    'random_state' :[7,42],
    'max_depth' :[10,50,100,1000],#決定木の深さ
    'min_samples_leaf': [10,50,100,1000],#分岐し終わったノードの最小サンプル数
    'min_samples_split': [10,50,100,1000]#決定木が分岐する際に必要なサンプル数
}

clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, iid=False)

clf.fit(xTrain, yTrain)

best_clf = clf.best_estimator_ #ここにベストパラメータの組み合わせが入っています
print('score: {:.2%}'.format(best_clf.score(xTrain, yTrain)))
yPred = clf.predict(xTest)
print('score: {:.2%}'.format(best_clf.score(xTest, yTest)))

print(classification_report(yTest, yPred))

fig = plt.figure()
mat = confusion_matrix(yTest, yPred)
sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')
plt.xlabel('predicted class')
plt.ylabel('true value')
fig.savefig("img.png")