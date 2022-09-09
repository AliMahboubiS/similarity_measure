import numpy as np
from math import *
from sklearn.metrics.pairwise import cosine_similarity

def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity_V0(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def cosine_similarity_V1(A, B):
    numerator = np.dot(A, B)
    denominator = sqrt(A.dot(A)) * sqrt(B.dot(B))
    return numerator / denominator


obj01 = [7, 1]
obj02 = [2, 10]
obj03 = [2, 20]

m = np.array(obj01)
b = np.array(obj02)
n = np.array(obj03)

print(cosine_similarity_V1(m, b))  # => 0.3328201177351375
print(cosine_similarity_V1(b, n))  # => 0.9952285251199801
print(cosine_similarity_V1(n, m))  # => 0.2392231652082992


# use sklearn lib without implementation the prodecure
m = np.array(obj01).reshape(1,-1)
b = np.array(obj02).reshape(1,-1)
n = np.array(obj03).reshape(1,-1)
print( cosine_similarity(m,b)[0,0] ) #=> 0.3328201177351375
print( cosine_similarity(b,n)[0,0] ) #=> 0.9952285251199801
print( cosine_similarity(n,m)[0,0] ) #=> 0.2392231652082992