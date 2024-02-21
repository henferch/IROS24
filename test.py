from operator import itemgetter
a = ['a','b','c','d','e']
b = itemgetter(0,4)(a)
print(b)