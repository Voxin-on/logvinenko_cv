import numpy as np
import matplotlib.pyplot as plt
import socket 
from skimage.measure import (label,
                            regionprops)

host="84.237.21.36"
port=5152

def recvall(sock,nbytes):
    data=bytearray()
    while len(data)<nbytes:
        packet=sock.recv(nbytes-len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

plt.ion()
plt.figure()
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
    sock.connect((host,port))
    sock.send(b"124ras1")
    print("Responce: ",sock.recv(10))

    i=0
    while i<10:
        sock.send(b"get")
        bts=recvall(sock,40002)

        im1=np.frombuffer(bts[2:],dtype="uint8")
        im1=im1.reshape(bts[0],bts[1])
        labeled=label(im1>0)
        if labeled.max()==2:
            pos1=np.unravel_index(np.argmax(np.where(labeled == 1, im1, 0)),im1.shape)
            pos2=np.unravel_index(np.argmax(np.where(labeled == 2, im1, 0)),im1.shape)
    
            result = np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))
            sock.send(f"{round(result,1)}".encode())
            print(sock.recv(10))

            i+=1