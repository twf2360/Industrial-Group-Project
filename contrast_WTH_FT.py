import math
import skimage
from skimage import io, viewer, color, data, filters, feature, morphology, exposure
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy import ndimage

'''
first, read the sample in 
'''
scan_gs_sample = np.load('scan1_75dpi.npy')



'''
then increase the contrast
'''
c_min, c_max = np.percentile(scan_gs_sample, (3,97)) #these are the paramters for the contrast
contrastedSample = exposure.rescale_intensity(scan_gs_sample, in_range=(c_min, c_max)) #returns the image with increased contrast 



'''
then, a white top hat transform to try to remove noise
returns Inv, which is *hopefully* a noise reduced version of the contrasted sample
'''
StructureElem = morphology.square(2) 
WhiteTophat = morphology.white_tophat(contrastedSample, selem=StructureElem)
Inv = contrastedSample - WhiteTophat


'''
take the fourier transform of the reduced noise
'''
sample_FT = fftpack.fft2(scan_gs_sample)
contrastedSample_FT = fftpack.fft2(contrastedSample)
inv_FT = fftpack.fft2(Inv)

'''
plot the fourier transform against the original image
'''

fig, ax = plt.subplots(ncols=3,nrows=2,figsize =(8,2.5))

ax[0][0].imshow(scan_gs_sample, cmap='gray')
ax[0][0].set(xlabel='', ylabel = '', title = 'Original Sample')

ax[0][1].imshow(contrastedSample, cmap='gray')
ax[0][1].set(xlabel='', ylabel = '', title = 'Higher Contrast Sample')

ax[0][2].imshow(Inv, cmap='gray')
ax[0][2].set(xlabel='', ylabel = '', title = 'Sample - WTH Transform')

ax[1][0].plot(np.arange(0,np.size(sample_FT[0]),1),np.abs(sample_FT[200]) )
ax[1][0].set(xlabel='pixel number', ylabel='FT', yscale = 'log',
       title='Fourier Transform of 200th row of pixels \n with a log scale')
ax[1][0].grid()


ax[1][1].plot(np.arange(0,np.size(contrastedSample_FT[0]),1),np.abs(contrastedSample_FT[200]) )
ax[1][1].set(xlabel='pixel number', ylabel='FT',
       title='Fourier Transform of 200th row of pixels with log scale \n of the higher contrast sample', yscale='log')
ax[1][1].grid()

ax[1][2].plot(np.arange(0,np.size(inv_FT[0]),1),np.abs(inv_FT[200]) )
ax[1][2].set(xlabel='Pixel Number', ylabel = 'FT after WTH', yscale='log',
title = 'Fourier Transform of 200th row of pixels \n after WTH transform, log scale')
ax[1][2].grid()


plt.show()

'''
not only does the final result have a lot of noise, it's like a fractal. everytime you cut a section out, the same pattern repeats. 
'''