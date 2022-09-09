from math import *

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def euclidean_dist_V0(doc1, doc2):
    '''
    For every (x,y) pair, square the difference
    Then take the square root of the sum
    '''
    pre_square_sum = 0
    for idx, _ in enumerate(doc1):
        pre_square_sum += (doc1[idx] - doc2[idx]) ** 2

    return sqrt(pre_square_sum)


def euclidean_dist_V1(x, y):
    '''
    This Version is very fast for vectorization
    without using for loop
    '''
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


# Example for 2D point and ND point
point1_2D = [0, 3]
point2_2D = [7, 6]

print(euclidean_dist_V0(point1_2D, point2_2D))
print(euclidean_dist_V1(point1_2D, point2_2D))
# => 7.615773105863909

# Example for 2D point and ND point
point1_ND = [0, 3, 8, -1]
point2_ND = [7, 6, 2, 0]

print(euclidean_dist_V0(point1_ND, point2_ND))
print(euclidean_dist_V1(point1_ND, point2_ND))
# => 9.746794344808963

# use sklearn lib without implementation the prodecure
t = np.array(point1_ND).reshape(1, -1)
n = np.array(point2_ND).reshape(1, -1)
print(euclidean_distances(t, n)[0][0])
# => 9.746794344808963
