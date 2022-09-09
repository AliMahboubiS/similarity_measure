import numpy as np

obj1 = [1,2,5]
obj2 = [1,3,5]
obj3 = [5,3,5]

def pearsons_correlation_coef(x, y):
    x = np.array(x)
    y = np.array(y)

    x_mean = x.mean()
    y_mean = y.mean()
    x_less_mean = x - x_mean
    y_less_mean = y - y_mean

    numerator = np.sum(xm * ym)
    denominator = np.sqrt(
        np.sum(xm ** 2) * np.sum(ym ** 2)
    )

    return r_num / r_den


pearsons_correlation_coef(obj1, obj2)
# => 0.9607689228305226