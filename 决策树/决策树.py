import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import sklearn
from sklearn.model_selection import train_test_split  

data=[]
labels=[]
with open('决策树_txt.txt') as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append ([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x=np.array(data)
labels=np.array(labels)
y=np.zeros(labels.shape)
y[labels=='fat']=1#二进制标签

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=tree.DecisionTreeClassifier(max_depth=3)#限制树深
clf.fit(x_train,y_train)
with open('tree.dot','w') as f:
    f=tree.export_graphviz(clf,out_file=f)
print(clf.feature_importances_)
answer=clf.predict(x_train)
print(x_train)
print(answer)
print(y_train)
print(np.mean(answer==y_train))
#准确率，召回率，f1得分
precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
answer=clf.predict_proba(x)[:,1]
print(classification_report(y,answer,target_names=['thin','fat']))