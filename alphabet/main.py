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

def symmetry(region,transponce=False):
    image = region.image
    if transponce:
        image=image.T
    shape =image.shape
    top = image[:shape[0]//2]
    if shape[0] % 2 != 0:
        bottom=image[shape[0]//2+1:]
    else:
        bottom=image[shape[0]//2:]
    bottom = bottom[::-1]
    result = bottom==top
    return result.sum()/result.size
                    
        
def classificator(region):
    holes = count_holes(region)
    if holes == 2:
        v, _ = count_lines(region)
        v /= region.image.shape[1]
        if v>0.2:
            return "B"
        else:
            return "8"
    elif holes == 1:
        img= region.image
        height, width = img.shape
        
        left_column_width = max(1, width // 7)
        left_edge = img[:, :left_column_width]
        
        density = left_edge.mean()

        if density > 0.8:
            if symmetry(region) > 0.8:
                return "D"
            else:
                return "P"
        else:
            if symmetry(region) > 0.7:
                return "0"
            else:
                return "A"
    elif holes == 0: 
        if region.image.sum()/region.image.size>0.95:
            return "-"
        shape = region.image.shape
        aspect = np.min(shape) / np.max(shape)
        if aspect>0.9:
            return "*"
        # v_asym = symmetry(region)
        # h_asym = symmetry(region, transponce=True)

        bg=np.logical_not(region.image) 
        labeled=label(bg) 
        if labeled.max()==4: 
            return "X" 
        elif labeled.max()==5: 
            return "W" 

        v, _ = count_lines(region)
        if v > 1:
            return "1"
        else:
            return "/"
    return "?"

image=imread("symbols.png")[:,:,:-1]
abinary=image.mean(2)>0
alabeled=label(abinary)
print(np.max(alabeled))
aprops=regionprops(alabeled)

result={}
image_path=save_path/"out_tree"
image_path.mkdir(exist_ok=True)

# plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol=classificator(region)
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