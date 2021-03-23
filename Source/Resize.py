# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:54:25 2018

@author: Amirhassan

bilinear interpolation to change size of image
This function is used to upsample the down-sampled image in image pyramid
main function that implements the image resizing is: Resize(img, out_dims)
img: the input image
out_dims: the desired output dimensions (can be larger or smaller than original image)
"""
import numpy as np
from PIL import Image

# read image as numpy array and normalize by 255
img = np.asarray(Image.open('butterfly.jpg').convert('L'))/255
# back to image im = PIL.Image.fromarray(numpy.uint8(I))

def sub2ind(array_shape, rows, cols): #this function return indce of element in an array
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind): # this reverse the above function
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def Resize(img, out_dims): # resize image k times
    # set dimensions of output image
    img_row = img.shape[0]
    img_col = img.shape[1]
    out_row = out_dims[0]
    out_col = out_dims[1]
    
    S_R = img_row / out_row; # porportion of original and resized image sizes
    S_C = img_col / out_col; # porportion of original and resized image sizes
    
    # create grid 
    cf, rf = np.meshgrid(np.linspace(1,out_col,out_col), np.linspace(1,out_row,out_row))
    rf = rf * S_R # aplly resizing scale
    cf = cf * S_C # aplly resizing scale
    r = np.floor(rf) # find number of rows
    c = np.floor(cf) # find number of columns
    
    # handle boundries !
    r[r < 1] = 1
    c[c < 1] = 1
    r[r > img_row -1] = img_row -1
    c[c > img_col -1] = img_col -1
    # difference of rows and coluns in mesh and original image
    delta_R = rf - r
    delta_C = cf - c
    # find indices using defined functions
    in1_ind = r + (c-1) * img.shape[0]
    in2_ind = (r+1) + (c-1) * img.shape[0]
    in3_ind = r + (c) * img.shape[0]
    in4_ind = r+1 + (c) * img.shape[0]
    
    # implement the bi-linear interpolation using indices of 4-neighbors
    temp = np.array(img.flatten(order='F')[in1_ind.astype(int)-1]) *(1 - delta_R) * (1 - delta_C) +\
    np.array(img.flatten(order='F')[in2_ind.astype(int)-1]) *(delta_R) * (1 - delta_C) +\
    np.array(img.flatten(order='F')[in3_ind.astype(int)-1]) *(1 - delta_R) * (delta_C) +\
    np.array(img.flatten(order='F')[in4_ind.astype(int)-1]) *(delta_R) * (delta_C)
    return temp

#res_img = Resize(img, 1)
#import matplotlib.pyplot as plt
#plt.imshow(res_img, cmap ='gray') # view LoG filtered Scaled image
#result = Image.fromarray((res_img * 255).astype(np.uint8))
#result.save('out.png')



    