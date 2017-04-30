'''
Author: Qiming Chen
Date Apr 30 2017
Description: A CUDA version using Shared memory to calculate the Mandelbrot set
Usage: 1. setup cuda environment 2. python mandelbrot_gpu.py
'''

from numba import cuda, int32, float64
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
    A GPU version (with using Shared Memory)of calculating the mandel value for each element in the 
    image array. The real and imag variables contain a 
    value for each element of the complex space defined 
    by the X and Y boundaries (min_x, max_x) and 
    (min_y, max_y).

    Shared memory: 
    The content in shared memory is shared only in one block (32 threads).
    By partitioning the problem, every 12 thread shares the same 'height', 'width', 'pixel_size' 'block' and so on in 1 block.
    This calculation can be finished in only 1 thread for 1 block instead of all thread in one block

    Also, caching results into shared memory and writing to global memory at the end can reduce the amount of reading and writing the global memory
    because reading/writing from/into consecutive memory location can actually reduce the i/o time.
    '''

    y, x = cuda.grid(2) #absolute thread id (y, x) in (1024, 128)

    s_shape = cuda.shared.array(shape=(2, 1), dtype=int32)
    s_pixel_size = cuda.shared.array(shape=(2, 1), dtype=float64)
    s_block = cuda.shared.array(shape=(2, 1), dtype=int32)

    if x == 0:
        grid_y, grid_x = cuda.gridsize(2) #(1024, 128) == > (1024, 128*12) for (1024, 1536)

        height, width = image.shape # 1024, 1536
        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = (max_y - min_y) / height
        block_y = height // grid_y # 1
        block_x = width // grid_x # 12

        s_shape[0] = height
        s_shape[1] = width

        s_pixel_size[0] = pixel_size_x
        s_pixel_size[1] = pixel_size_y

        s_block[0] = block_x
        s_block[1] = block_y

    cuda.syncthreads()

    s_image = cuda.shared.array(shape=(s_block[1][0],s_block[0][0]), dtype=int32)

    y, x = cuda.grid(2) 

    for i in range(s_block[0][0]):
        thread_x = x * s_block[0][0] + i
        real = min_x + thread_x * s_pixel_size[0]
        for j in range(s_block[1][0]):
            thread_y = y * s_block[1][0] + j
            imag = min_y + thread_y * s_pixel_size[1]
            if thread_y < s_shape[0] and thread_x < s_shape[1]:
                s_image[j, i] = mandel(real, imag, iters)

    for i in range(s_block[0][0]):
        thread_x = x * s_block[0][0] + i
        for j in range(s_block[1][0]):
            thread_y = y * s_block[1][0] + j
            if thread_y < s_shape[0] and thread_x < s_shape[1]:
                image[thread_y, thread_x] = s_image[j,i]

    cuda.syncthreads()

    if x == 0:
        print("thread(", y, "/1024)" ":", " done")
    
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    threadsperblock = (32, 8)
    blockspergrid = (32, 16) # (1024,128) ==> (1024, 128*12)

    image_global_mem = cuda.to_device(image)
    compute_mandel[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()