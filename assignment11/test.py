'''
Author: Qiming Chen
Date: Apr 23 2017

Description: Tester for parallel_sorter.py
1. create an array with random integers
2. call parallel_sorter for sorting
3. the length of the result and the order of that sorted array is checked for correctness

mpiexec -n <number of process> python test.py 
'''

import parallel_sorter
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Testing
array_size = 10000
data = np.random.randint(low=0, high=array_size, size=array_size)
data_sorted = parallel_sorter.parallel_sort(array_size, data)

# Verify the increasing order of the sorted array in root process -- check the order and track the error if not sorted
if rank == 0:
	# check the length
	print("Same length:", len(data_sorted)==array_size) 
	# check the order
	print("Sorted:", np.alltrue(data_sorted == sorted(data)))