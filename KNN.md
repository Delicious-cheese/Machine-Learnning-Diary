# What is KNN?
概念：K近邻算法（最近邻居法）
![image](https://github.com/ccttvv/Machine-Learnning-Diary/blob/master/images/intro_KNN.png)
<br/>
如上图，在一堆点中，判断p是属于红色分类还是蓝色分类
在这里我们选三个与p点距离相对短的点，这三个点都属于红色分类，预测p是属于红色分类

# How to use it?
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# 1.引入数据
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)


# 2.训练模型
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

# 3.预测
y_predict = knn_clf.predict(x)

# 4.准确率
score = knn_clf.score(X_test, y_test)

```



