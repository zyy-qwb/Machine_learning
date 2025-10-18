import numpy as np
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()#取得knn分类器
data=np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])#sklearn库要求树如二维数组，故[[]]
labels=np.array([1,1,1,2,2,2])
knn.fit(data,labels)
pre=knn.predict([[18,90]])
print(pre)
