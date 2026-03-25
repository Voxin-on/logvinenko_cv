import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import (label,
                            regionprops)

trajectories = {}


for i in range(100):
    image=np.load(f"out/h_{i}.npy")
    labeled=label(image)
    regions = regionprops(labeled)

    if i == 0:
        for j, region in enumerate(regions):
            trajectories[j] = [region.centroid]
    else:
        for region in (regions):
            cur_cord = region.centroid

            min_num=0
            min_dist = float('inf')

            for num,cord in trajectories.items():
                last_cord=cord[-1]
                dist=abs(cur_cord[0]-last_cord[0])+abs(cur_cord[1]-last_cord[1])

                if min_dist>dist:
                    min_dist=dist
                    min_num=num

            trajectories[min_num].append(cur_cord)



for num, obj in trajectories.items():
    cord = np.array(obj)
    plt.plot(cord[:, 1], cord[:, 0], marker='o', label=num)

plt.legend()
plt.show()