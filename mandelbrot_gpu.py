'''
Author: Qiming Chen
Date Apr 30 2017
Description: A CUDA version to calculate the Mandelbrot set
Usage: 1. setup cuda environment 2. python mandelbrot_gpu.py
'''

from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    A GPU version of calculating the mandel value for each element in the 
    image array. The real and imag variables contain a 
    value for each element of the complex space defined 
    by the X and Y boundaries (min_x, max_x) and 
    (min_y, max_y).

    Step 1:  define the absolute thread id (y, x) in (1024, 1536) and 
    Step 2:  define task size (e.g (1,12)) of each thread such to assign (1024 1536) tasks into (1024, 128) threads
    Step 3:  finish tasks in each thread
    '''
    grid_y, grid_x = cuda.gridsize(2) #(1024, 128) 
    y, x = cuda.grid(2) # (y, x) where y is in [0, 1023] and x is in [0, 127]
    height, width = image.shape # 1024, 1536

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    # get the partition index of the y and x
    block_y = height // grid_y # 1
    block_x = width // grid_x # 12

    # every thread in (1024, 128) should handle (1, 12) tasks such that the totally (1024, 1536) tasks are evenly assigned

    for i in range(block_x):
        thread_x = x * block_x + i
        real = min_x + thread_x * pixel_size_x
        for j in range(block_y):
            thread_y = y * block_y+ j
            imag = min_y + thread_y * pixel_size_y
            if thread_y < height and thread_x < width:
                image[thread_y, thread_x] = mandel(real, imag, iters)
    
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    threadsperblock = (32, 8)
    blockspergrid = (32, 16) # (1024,128) ==> (1024, 128*12)

    image_global_mem = cuda.to_device(image)
    compute_mandel[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()