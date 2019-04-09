# What is KNN?
概念：K近邻算法（最近邻居法）
![image](https://github.com/ccttvv/Machine-Learnning-Diary/blob/master/images/intro_KNN.png)
<br/>
如上图，在一堆点中，判断p是属于红色分类还是蓝色分类
在这里我们选三个与p点距离相对短的点，这三个点都属于红色分类，预测p是属于红色分类

# How to use it?
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. 数据
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# 2.转成numpy.array 方便矩阵运算， 并添加要预测的点x

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514])

# 找3个相近的点
kNN_clf = KNeighborsClassifier(n_neighbors=3)

# 训练模型
kNN_clf.fit(X_train, y_train)

# 预测
y = kNN_clf.predict(x.reshape(1,-1))

```

# Implement


