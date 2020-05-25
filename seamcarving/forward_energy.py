import numpy as np
import numba as nb
import cv2


@nb.jit
def forward_energy(image):
    h, w, c = image.shape
    m = np.zeros((h, w))
    energy = np.zeros((h, w))
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for i in range(1, h):
        for j in range(w):
            up = (i - 1) % h
            down = (i + 1) % h
            left = (j - 1) % w
            right = (j + 1) % w
            
            cU = np.abs(I[i, right] - I[i, left])
            cL = np.abs(I[up, j] - I[i, left]) + cU
            cR = np.abs(I[up, j] - I[i, right]) + cU
            
            mU = m[up, j]
            mL = m[up, left]
            mR = m[up, right]
            
            cULR = np.array([cU, cL, cR])
            mULR = np.array([mU, mL, mR]) + cULR
            
            x = np.argmin(mULR)
            m[i, j] = mULR[x]
            energy[i, j] = cULR[x]
    
    return energy
