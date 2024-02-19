import numpy as np


def dtw_distance(x, y, loss_f=None):
    """
    calculate the dtw algorithm distance of two different length time series
    :param x: times series value, list,np.array,pd.Series
    :param y: times series value, list,np.array,pd.Series
    :param loss_f: loss function
    :return: value of distance
    """
    # delete Na,NaN,Inf
    x = np.array(x)
    x = x[np.isfinite(x)]
    y = np.array(y)
    y = y[np.isfinite(y)]

    # default loss function
    if loss_f is None:
        loss_f = lambda a, b: (a - b) * (a - b)

    row = len(x)
    column = len(y)

    if row > column:
        temp1 = [np.infty] * (column + 1)
        temp1[0] = 0
        for i in range(row):
            temp2 = [np.infty] * (column + 1)
            for j in range(column):
                temp2[j + 1] = loss_f(x[i], y[j]) + min(temp2[j], temp1[j], temp1[j + 1])
            temp1 = temp2
    else:
        temp1 = [np.infty] * (row + 1)
        temp1[0] = 0
        for i in range(column):
            temp2 = [np.infty] * (row + 1)

            for j in range(row):
                temp2[j + 1] = loss_f(x[j], y[i]) + min(temp2[j], temp1[j], temp1[j + 1])
            temp1 = temp2

    return temp1[-1]
