#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.io as sio
from matplotlib.axes import Axes as axes


# In[7]:


def read(file):
    return cv2.imread(file,0)


# In[80]:


def show_image(img,xlim = None,ylim = None):
    if xlim != None and ylim != None:
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
    plt.imshow(img,cmap = 'gray')
    


# In[38]:


def fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


# In[4]:


def mag(f):
    magnitude_spectrum = 20*np.log(np.abs(f)+1)
    return magnitude_spectrum


# In[22]:


def inv_fourier(f):
    f_ishift = np.fft.ifftshift(f)
    f_i = np.fft.ifft2(f_ishift)
    return abs(f_i)


# In[67]:


def sinusoid_noise(x,y,size):
    #size = [256,256]
    #y = 10
    #x =5
    noise = fourier(np.ones(size))
    y_c = size[0]//2 
    x_c = size[1]//2 
    noise[y_c-y,x_c-x] = noise[y_c,x_c]/2 # y rows and x columns away from center
    noise[y_c+y,x_c+x] = noise[y_c,x_c]/2
    noise_t = inv_fourier(noise)
    return noise_t


# In[60]:


#size = [256,256]
#y = 10
#x =5
#a = sinusoid_noise(x,y,size)
#show_image(a)


# In[84]:


def phase(f):
    ph = np.exp(1j*np.angle(f))
    return ph


# In[87]:


def concat(img1,img2):
    c = np.concatenate((img1,img2),axis=1)
    return c

