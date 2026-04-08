import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
from skimage.io import imread
from pathlib import Path

save_path=Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0]+2,shape[1]+2))
    new_image[1:-1,1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled)-1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image,0)/shape[0]==1).sum()
    hlines = (np.sum(image,1)/shape[1]==1).sum()
    return vlines,hlines

def horizontal_symmetry(region):
    img = region.image.astype(int)
    flipped = np.fliplr(img)
    diff = np.abs(img - flipped).sum() /region.image.size
    return diff

def extractor(region):
    cy,cx=region.centroid_local
    cy/=region.image.shape[0]
    cx/=region.image.shape[1]
    perimetr = region.perimeter / region.image.size
    holes = count_holes(region)
    v,h = count_lines(region)
    v /= region.image.shape[1]
    h /= region.image.shape[0]
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] /region.image.shape[1]
    
    solidity = region.solidity
    symmetry = horizontal_symmetry(region)

    return np.array([region.area/region.image.size,cy,cx,perimetr,
            holes,v,h,eccentricity,aspect,solidity,symmetry])

def classificator(region,templates):
    features=extractor(region)
    result=""
    min_d=10**16
    for symbol,t in templates.items():
        d=((t-features)**2).sum()** 0.5
        if d<min_d:
            result=symbol
            min_d=d
    return result

template=imread("alphabet-small.png")[:,:,:-1]
print(template.shape)
template=template.sum(2)
binary=template!=765

labeled=label(binary)
props=regionprops(labeled)
print(type(props[0]))
print(props[0].area,props[0].centroid)

templates={}

for region,symbol in zip(props,["8","O","A","B","1","W","X","*","/","-"]):
    templates[symbol]=extractor(region)
    
print(templates)

print(classificator(props[0],templates))

image=imread("alphabet.png")[:,:,:-1]
abinary=image.mean(2)>0
alabeled=label(abinary)
print(np.max(alabeled))
aprops=regionprops(alabeled)

result={}
image_path=save_path/"out"
image_path.mkdir(exist_ok=True)

plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol=classificator(region,templates)
    if symbol not in result:
        result[symbol]=0
    result[symbol]+=1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path/f"image_{region.label}.png")
print(result)

plt.imshow(abinary)
plt.show()