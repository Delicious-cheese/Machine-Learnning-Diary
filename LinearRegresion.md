

from mySklearn.SimpleLinearRegresion import SimplerLinearRegresion1
import numpy as np

# 使用自身实现的LinearRegresion
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
reg = SimplerLinearRegresion1()
reg.fit(x, y)
print(reg.a_)
print(reg.b_)
print(reg.predict(np.array([1,2])))
