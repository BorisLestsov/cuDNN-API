import numpy as np
import matplotlib.pyplot as plt

f = open("kernels.dat", 'rb')

meta = np.fromfile(f, dtype=np.int32, count=4)
print meta

kers = np.fromfile(f, dtype=np.float32, count=meta[0]*meta[1]*meta[2]*meta[3])
print kers.shape
kers = kers.reshape((meta[0],meta[1],meta[2],meta[3]))
print kers.shape
kers = kers.swapaxes(0, 1)
kers = kers.swapaxes(1, 2)
kers = kers.swapaxes(2, 3)
print kers.shape

im0 = kers[0, :, :, :]
plt.imshow(im0, interpolation='none')
plt.show()
