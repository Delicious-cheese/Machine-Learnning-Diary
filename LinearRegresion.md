
```python
from mySklearn.SimpleLinearRegresion import SimplerLinearRegresion1, SimpleLinearRegresion2
import numpy as np
import timeit

reg1 = SimplerLinearRegresion1()
'''
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
reg1.fit(x, y)
print(reg.a_)
print(reg.b_)
print(reg.predict(np.array([1,2])))
'''

reg2 = SimpleLinearRegresion2()


# 比较循环和向量话
# 生成100万个(x,y) 其中x取值[0,1)
n = 1000000
x = np.random.random(size=n)
y = x * 1 + 2 + np.random.normal(size=n)

'''
jupertnote 使用
%timeit reg1.fit(x, y)
%timeit reg2.fit(x,y)
'''
```
