import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (opening,
                                dilation,
                                closing,
                                erosion)

image=np.load("wires/wires5.npy")
struct=np.ones((3,1))
processed=opening(image, footprint=struct)
labeled=label(image)

for n in range(1,labeled.max()+1):
    wire=(labeled==n)
    processed_wire=opening(wire, footprint=struct)
    count=label(processed_wire)
    print(f"Wire: {n}, parts={count.max()}")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(processed)
plt.show()