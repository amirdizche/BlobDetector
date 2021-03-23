# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:08:12 2018
Modified to increase speed 
@author: afallah

main function that calculates the  2D convolution is :
    conv2d(image, kernel, shape, padding)
    image:  the input image
    kernel: the desired filter
    shpe: {'full', 'same', 'valid'} choose one option
    padding: {'zero', 'wraparound', 'copyedge', 'reflected'} choose one option
"""

import numpy as np

# Define Kernels
down2 = np.array([[0.25,0.25],[0.25,0.25]])
sobel = (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
laplacian = (np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
laplacian2 = (np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
gaussian = (np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
sharpen = (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
identity = (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))

#convolution function
def conv2d(image, kernel, shape = 'same', padding = 'zero'):

#    # normalize kernel
#    if abs(kernel.sum()) > 10^-3:
#        kernel = kernel / kernel.sum()
    
    # dimensions of image
    i_x, i_y = image.shape[0], image.shape[1]
    # dimensions of kernel
    k_x, k_y = kernel.shape[0], kernel.shape[1]
    
    output = []
    # output size
    if shape == 'full': # read shape format
        output=np.zeros([i_x+2,i_y+2]) # calculate output size
        
        if padding == 'zero': # read padding format
            
            padded_image=np.pad(image, (2, 2), 'constant', constant_values=0)# pad image
            # iterate over pixels
            for i in range(np.shape(output)[0]):
                for j in range(np.shape(output)[1]):
                    temp=padded_image[i:i+k_x,j:j+k_y] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
        elif padding == 'wraparound': # read padding format
            
            padded_image=np.pad(image, (2, 2), 'wrap') # pad image
            # iterate over pixels
            for i in range(np.shape(output)[0]):
                for j in range(np.shape(output)[1]):
                    temp=padded_image[i:i+k_x,j:j+k_y] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
        elif padding == 'copyedge': # read padding format
            
            padded_image=np.pad(image, (2, 2), 'edge')# pad image
            # iterate over pixels
            for i in range(np.shape(output)[0]):
                for j in range(np.shape(output)[1]):
                    temp=padded_image[i:i+k_x,j:j+k_y] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
        elif padding == 'reflected': # read padding format
            
            padded_image=np.pad(image, (2, 2), 'reflect', constant_values=0)# pad image
            # iterate over pixels
            for i in range(np.shape(output)[0]): 
                for j in range(np.shape(output)[1]):
                    temp=padded_image[i:i+k_x,j:j+k_y] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
    elif shape == 'same':  # read shape format
        output=np.zeros([i_x,i_y])  # calculate output size
        
        if padding == 'zero': # read padding format
            
            padded_image=np.zeros([i_x+k_x-1,i_y+k_y-1])# pad image
            padded_image[int((k_x-1)/2):(i_x+int((k_x-1)/2)),int((k_y-1)/2):i_y+int((k_y-1)/2)]=image
            # iterate over pixels
            for i in range(i_x):
                for j in range(i_y):
                    temp=padded_image[i:i+k_x,j:j+k_x] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
        elif padding == 'copyedge': # read padding format
            
            padded_image=np.pad(image, (int((k_x-1)/2), int((k_y-1)/2)), 'edge')# pad image
            # iterate over pixels
            for i in range(i_x):
                for j in range(i_y):
                    temp=padded_image[i:i+k_x,j:j+k_x] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output 
        
        elif padding == 'wraparound': # read padding format
            
            padded_image=np.pad(image, (int((k_x-1)/2), int((k_y-1)/2)), 'wrap')# pad image
            # iterate over pixels
            for i in range(i_x):
                for j in range(i_y):
                    temp=padded_image[i:i+k_x,j:j+k_x] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
        elif padding == 'reflected': # read padding format
            
            padded_image=np.pad(image, (int((k_x-1)/2), int((k_y-1)/2)), 'reflect')# pad image
            # iterate over pixels
            for i in range(i_x):
                for j in range(i_y):
                    temp=padded_image[i:i+k_x,j:j+k_x] # find neighborhood
                    output[i][j]=sum(sum(kernel*temp)) # multiply filter to neiborhood and sum
            return output
        
    elif shape == 'valid':  # read shape format
        
        output = np.zeros([i_x - 2 * k_x, i_y - 2 * k_y])  # calculate output size
        i_x, i_y = output.shape[0], output.shape[1]
        image_pad = image[3:i_x+3,3:i_y+3]
        # iterate over pixels
        for j in range(i_x):
            for i in range(i_y):
                fitered_sum = 0
                for jk in range(k_y):
                    for ik in range(k_x):
                        p = 0
                        if (j + jk -1 >= 0) and (j + jk -1 < i_x) and (i + ik -1 >= 0) and (i + ik -1 < i_y):
                            p = image_pad[j + jk -1, i + ik -1] # find neighborhood
                        weight = kernel[jk, ik] # multiply filter to neiborhood
                        fitered_sum += p * weight # sum the values
                output[j, i] = fitered_sum
        return output



                    