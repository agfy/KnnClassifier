import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.preprocessing import scale

data = pd.read_csv('wine.data')
X = np.empty([len(data['1']), 13])
y = np.empty(len(data['1']))
for index, row in data.iterrows():
    y[index] = row['1']
    X[index][0] = row['14.23']
    X[index][1] = row['1.71']
    X[index][2] = row['2.43']
    X[index][3] = row['15.6']
    X[index][4] = row['127']
    X[index][5] = row['2.8']
    X[index][6] = row['3.06']
    X[index][7] = row['.28']
    X[index][8] = row['2.29']
    X[index][9] = row['5.64']
    X[index][10] = row['1.04']
    X[index][11] = row['3.92']
    X[index][12] = row['1065']

X = scale(X)
k_fold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
k_max = 0
summ_max = 0.0

for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    score_arr = model_selection.cross_val_score(neigh, X, y, cv=k_fold)
    summ = 0.0
    for val in score_arr:
        summ += val

    summ /= len(score_arr)
    if summ > summ_max:
        summ_max = summ
        k_max = k
