import numpy as np
from skimage.measure import label
from skimage.morphology import opening

image=np.load("stars.npy")

plus = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

cross = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])

pluses=label(opening(image,plus))
crosses=label(opening(image,cross))

count=pluses.max()+crosses.max()

print(count)