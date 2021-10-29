# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:30:19 2021

@author: kntl
"""
#import cv2
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#EKSTRASI FITUR
def ekstraksi_fitur(Z, threshold=50):
    #Z = np.array(Z)
    #Z=Image.open(Z).convert('L')
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]



"""
def training(datasets):
    fitur_datasets = []
    i = 0
    for data in datasets:
        fitur_datasets.append(ekstraksi_fitur(data))
        print("data : ",data," selesai")        
        i+=1
    X = np.vstack(fitur_datasets)
    return X
"""