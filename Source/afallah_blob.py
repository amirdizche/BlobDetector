# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:20:02 2018
@author: Amirhassan

BLOB DETECTOR

"""
# import required libraris and functions
import numpy as np 
from PIL import Image
from Conv_2D import conv2d # The conv2d function that I desinged earlier
from Resize import Resize # The bi-linear interpolation function that I developed for this project
import time # to calculate computation time



gaussian = (np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) *  (1/9) # gassuan blurr filter for sub-sampling

# The Laplacian of Gaussian filter (LOG)
def LoG_filter(sigma):
    # order of filter (odd)
    n = int(2 * np.ceil(sigma * 2.5) + 1)    
    Log = np.zeros([n,n]) # zeors matrix with size of filter output
    for i in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
        for j in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
            Log[i + int(np.floor(n/2)), j +int(np.floor(n/2))] = \
            -(1/(np.pi * sigma ** 4)) * (1-((i**2 + j**2)/(2*sigma**2))) *\
            np.exp(-((i**2 + j**2)/(2*sigma**2))) * (sigma**2) # calculate each cell value based on LOG formula
    return Log

# The difference of Gassian filter (DOG) that approximates LOG but it is faster! (1.6*simga, sigma)
def Dog_filter(sigma):
    n = int(2 * np.ceil(sigma * 2.5) + 1) # order of filter (odd)  
    Dog1 = np.zeros([n,n]) # 1st Gaussuan
    Dog2 = np.zeros([n,n]) # 2nd Gaussian
    ss1 = 0
    ss2 = 0
    #calcualte 1st Gaussian based on gaussain formula
    for i in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
        for j in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
            ss1 = ss1 + np.exp(-(i**2 + j**2)/(2* 1.6 * sigma**2)) 
            
    for i in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
        for j in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
            Dog1[i + int(np.floor(n/2)), j +int(np.floor(n/2))] = \
            np.exp(-(i**2 + j**2)/(2* 1.6 * sigma**2)) /ss1
            
    #calcualte 2nd Gaussian based on gaussain formula
    for i in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
        for j in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
            ss2 = ss2 + np.exp(-(i**2 + j**2)/(2 * sigma**2)) 
            
    for i in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
        for j in range(int(-np.floor(n/2)),int(np.floor(n/2))+1):
            Dog2[i + int(np.floor(n/2)), j +int(np.floor(n/2))] = \
            np.exp(-(i**2 + j**2)/(2* sigma**2)) /ss2   
    
    Dog = 1.2 * (Dog1 - Dog2) * sigma **2 # subtract Gassuans and multipy by scale
    return Dog
    

def max_replace2(array, n): # implimentation of non-max supression
    rows = array.shape[0] # get simensions of input
    cols = array.shape[1]
    res_arr = np.zeros([rows, cols])
    for i in range(rows): # iterate over pixels
        for j in range(cols):
            if i-(n-1)/2 < 0: # handle boundaries
                a = 0
            else:
                a = int(i-(n-1)/2)  
            if j-(n-1)/2 < 0:  # handle boundaries
                b = 0
            else: 
                b = int(j-(n-1)/2)
            if i+1+(n-1)/2 > rows:  # handle boundaries
                a1 = int(rows)
            else:
                a1 = int(i+1+(n-1)/2)
            if (j+1+(n-1)/2) > cols:  # handle boundaries
                b1 = int(cols)
            else:
                b1 = int(j+1+(n-1)/2)
            neighb = array[a:a1, b:b1]  # create neigborhood
            res_arr[i, j] = np.max(neighb) # reolace with max value
    return res_arr
            
            
# function to create scale-space by down-sampling the image to be more efficient    
# create image pyramid and apply filter on them and svae the results in different scalse        
def scale_space_create(img, sigma, n, Filter = 'DOG'): # n is number of scales, sigma is the initial scale
    # Filter = 'DOG' or 'LOG'
    k=2**(0.35) #value of the scaling factor
    [org_row, org_col] = np.shape(img) # dimensions of input image
    scale_space = np.zeros([img.shape[0], img.shape[1], n]) # to store scalse-space
    Scale_Space = np.zeros([img.shape[0], img.shape[1], n]) # to scale max-replaces scalse-space
    for i in range(n):
        if Filter == 'LOG': # check whether to use LOG or DOG filter
            Log = LoG_filter(sigma) # create LOG filter
        else:
            Log = Dog_filter(sigma) # create DOG filter
        # now down sample blurred image
        out_row = int(np.floor(org_row * 1/k**(i)))
        out_col = int(np.floor(org_col * 1/k**(i)))
        image_down = Resize(img, [out_row, out_col])
        # apply LOG (DOG) filter
        Log_filtered = conv2d(image_down, Log, 'same', 'copyedge')
        Log_filtered = np.power(Log_filtered, 2)
        # save result to one slot of scale space
        scale_space[:,:,i] = Resize(Log_filtered, [org_row, org_col])
        # max replacement filter and save the result 
        Scale_Space[:,:,i] = max_replace2(scale_space[:,:,i], 7)
        # blurr for next downsampling
        img = conv2d(img, gaussian, 'same', 'copyedge')
#        sigma = sigma * k   # increase the scale 
    return scale_space, Scale_Space

# function to find the scale that has maximum response to filter
def find_scale(org_row, org_col, Scale_Space): 
    #inputs are origianl image size and max replaced scale-space
    max_scale = np.zeros(Scale_Space.shape)
    for i in range(org_row): # iteate over pixels
        for j in range(org_col):
           ind = np.unravel_index(np.argmax(Scale_Space[i,j,:]), Scale_Space[i,j,:].shape) # find max pixel indice
           max_scale[i,j,ind[0]] =  Scale_Space[i,j,ind[0]] # store scale and max vlues
    return max_scale


# find circles the center and radius of them
def find_circles(max_scale, threshold, sigma):
    radius = []
    coord = []
    k=2**(0.35)
    for i in range(max_scale.shape[2]): # compare to threshold
        ind = np.argwhere(max_scale[:,:,i] >= threshold)
        radius = np.ones([ind.shape[0],1]) * (2 ** 0.5 * sigma * k**(i)) # calculate radius based on scale 
        coord.append(np.concatenate((ind, radius), axis = 1)) # save coords of centers
    cc = np.zeros([1,3])
    for i in range(np.shape(coord)[0]):
        cc = np.concatenate((cc,coord[i]),axis = 0) # add circles info to an array to draw later
    return cc[1:,:]


# function to draw blocs given center coordinates, radius, and line color
def draw_circles(image, output_name, cx, cy, rad, color='r'):
    # inputs are image, center x coord, center y coord, radius of circle, and its color
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots() # create figure
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray') # set color map
    for x, y, r in zip(cx, cy, rad): # iterate over center and radius if circles
        circ = Circle((x, y), r, color=color, fill=False, linewidth = 1.1) # draw circle
        ax.add_patch(circ) # show the resulting circle
    plt.title('%i blobs found' % len(cx)) # title of the plot
    out_name = output_name + '.png'
    plt.savefig(out_name) # save the results as png
    plt.show()




def blob_detector(File_name, sigma, n, threshold, Filter = 'DOG', output_name = 'blob_out'):
    """
This is the main function that combines all above functions and implmenet blob detector
        blob_detector(image, sigma, n, threshold)
        It will show and save the result as .png file with given name
Input arguments:
    * File_name: name of the input image. Example 'butterfly.jpg'
      If the image is not in the same folder you can give the directory address & name
      example: 'C:/Users/Amirhassan/Documents/butterfly.jpg'
    * sigma: initial scale value
    * n: number of iterations to change scale
    * threshold: the threshold for detecting blobs. 
    * Filter: 'LOG', or 'DOG' 
    * output_name: desired name for the output, Example: 'butterfly_blob'
Example:
    blob_detector('butterfly.jpg', 2, 5, 0.05, 'DOG', 'butterfly_blob')
    """
    image = np.asarray(Image.open(File_name).convert('L'))/255 # load an input image and normalize it
    org_row, org_col = image.shape # get the dimensions of the image
    start = time.time() # start counting time
    # find scale-space and non-max supression
    scale_space, Scale_Space = scale_space_create(image, sigma, n, Filter)
    # find-max-replaced array
    max_scale = find_scale(org_row, org_col, Scale_Space)
    max_scale = np.multiply(max_scale, (max_scale == scale_space)) 
    # find circle coord and redius
    circles = find_circles(max_scale, threshold, sigma)
    # draw circles
    draw_circles(image, output_name, circles[:,1], circles[:,0], circles[:,2], color='r')
    end = time.time() # end counting time
    print('computation time = {:0.4f}'.format(end - start)) # print computation time

# one example
# uncomment following line to see the results or run it in console    
blob_detector('butterfly.jpg', 2, 10, 0.01, 'DOG', 'butterfly_blob_test')


