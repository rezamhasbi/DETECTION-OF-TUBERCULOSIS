# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 01:48:57 2021

@author: kntl
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import time
waktu = time.strftime("%Y%m%d-%H%M%S")
# print (waktu)

# C:\Users\Viony\cobaspider\normal

#---------------pre-processing------------------#
def pre_pro(dt_citra):
    i = 0
    for data in dt_citra:
        img = cv.imread(data)
        # konversi ukuran citra menjadi 512x512 px
        pjg = 512
        lbr = 512
        resz = cv.resize(img,(pjg,lbr))
        # konversi citra ke grayscale
        gray = cv.cvtColor(resz, cv.COLOR_BGR2GRAY)
        # menghilangkan noise dg median filter
        #median = cv.medianBlur(gray,7)
        equ = cv.equalizeHist(gray) #Adaptive histogram
        #blur = cv.GaussianBlur(equ,(5,5),0, borderType=cv.BORDER_CONSTANT) #gausian blur 
        
        h_th3, h_mask, hasil,autocanny = segmentasi(equ, resz)
        folder = r"C:\Users\kntl\OneDrive\Documents\BelajarPC\skripsi\hasil"
        #folder = r"C:\Users\kntl\Documents\BelajarPC\skripsi\hasil" # hasil segmentasi di simpan
        nama_file =  "hasil-%s.png" % waktu
        path = "%s/%s" % (folder,nama_file)
        cv.imwrite(os.path.join(folder,nama_file), hasil)
        print ("gambar ", data," jadi data ",i)
        i +=1
        print("path : ", path)
    return (hasil, path, h_th3, h_mask, equ,autocanny)

#---------------segmentasi------------------#
def segmentasi(citra, resz):
    
    # Otsu's thresholding 
    ret3,th3 = cv.threshold(citra,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #Autocanny edge
    sigma = 0.3
    median =np.median(th3)
    lower =int(max(0, (1.0 - sigma) * median))
    upper =int(min(255, (1.0 + sigma) * median))
    autocanny = cv.Canny(th3, lower, upper)
    
    # erosi, dilasi, closing
    kernel = np.ones((9,9),np.uint8)
    erosi = cv.erode(th3,kernel,iterations = 1)
    # dilasi = cv.dilate(erosi,kernel,iterations = 1)
    closing = cv.morphologyEx(erosi, cv.MORPH_CLOSE, kernel)
    mask = closing.copy()
    # flood fill to remove mask at borders of the image
    h, w = resz.shape[:2]
    for row in range(h):
        if mask[row, 0] == 255:
            cv.floodFill(mask, None, (0, row), 0)
        if mask[row, w-1] == 255:
            cv.floodFill(mask, None, (w-1, row), 0)
    for col in range(w):
        if mask[0, col] == 255:
            cv.floodFill(mask, None, (col, 0), 0)
        if mask[h-1, col] == 255:
            cv.floodFill(mask, None, (col, h-1), 0)
    # flood fill background to find inner holes
    holes = mask.copy()
    cv.floodFill(holes, None, (0, 0), 255)
    # invert holes mask, bitwise or with mask to fill in holes
    holes = cv.bitwise_not(holes)
    mask = cv.bitwise_or(mask, holes)
    # convex hull
    contours, hierarchy= cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        hull= cv.convexHull(cnt)
        cv.drawContours(mask, [hull],0,(255,255,255),-1)
    
        
    # final mask
    res_img = cv.bitwise_and(autocanny, autocanny, mask=mask)
    
    return th3, mask, res_img,autocanny


#datasets = glob.glob("../cobaspider/1-normal/n-1.jpeg")
# print("data : ", datasets)
# hasil, path, th3, mask = pre_pro(datasets)

# namaci = '../cobaspider/normal/2-.jpeg'

# c_prepro = pre_pro(resz)
# c_segmentasi = segmentasi(c_prepro, resz)
# print("citra : ", namaci)
# cv.imwrite(os.path.join(path, "normal.png"), res_img)

# plt.subplot(131), plt.imshow(th3, 'gray'),plt.title('Otsu')
# plt.subplot(132), plt.imshow(mask, 'gray'),plt.title('Convex Hull')
# plt.subplot(133), plt.imshow(hasil, 'gray'),plt.title('Result')


# plt.subplot(234), plt.imshow(mask, 'gray'),plt.title('convex hull')
# plt.subplot(235), plt.imshow(res_img, 'gray'),plt.title('open')
# plt.subplot(236), plt.imshow(final, 'gray'), plt.title('region')

# plt.imshow(coba)