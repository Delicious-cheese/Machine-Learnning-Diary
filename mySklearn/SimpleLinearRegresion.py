import numpy as np

# for 计算
class SimplerLinearRegresion1:

    # 初始化最小二乘法的参数

    def __init__(self):
        self.a_ = None
        self.b_ = None

    # 计算a和b的值
    def fit(self, X_train, y_train):
        assert X_train.ndim == 1, \
        "一维数据"
        assert len(X_train) == len(y_train), \
        "特征数量和标签数相同"

        # 最小二乘法

        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        sum = 0.0
        d = 0.0
        for x_i, y_i in zip(X_train, y_train):
            sum += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2


        self.a_ = sum / d
        self.b_ = y_mean - x_mean * self.a_

        return self

    # 预测
    def predict(self, x_predict):

        assert x_predict.ndim == 1, \
        "测试数据也是一维的"
        assert self.a_ is not None and self.b_ is not None, \
        "a 和 b 是已经计算好的"

        return np.array([self._predict(i) for i in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_

    def __repr__(self):
        return "simplelinerregresion1"

# 向量计算
class SimpleLinearRegresion2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, y_train):
        assert X_train.ndim == 1, \
            "一维数据"
        assert len(X_train) == len(y_train), \
            "特征数量和标签数相同"
        x_mean = np.array(X_train)
        y_mean = np.array(y_train)

        self.a_ = (X_train - x_mean).dot(y_train - y_mean) / (X_train - x_mean).dot(X_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "测试数据也是一维的"
        assert self.a_ is not None and self.b_ is not None, \
            "a 和 b 是已经计算好的"
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_

    def __repr__(self):
        return "SimpleLinearRegresion"
