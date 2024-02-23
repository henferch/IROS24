from operator import itemgetter
a = ['a','b','c','d','e']
b = itemgetter(0,4)(a)
print(b)

import numpy as np

b = np.array([1,0])
a = np.array([[1,0],[2.3]])

print(a==b)