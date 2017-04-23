'''
Author: Qiming Chen
Date: Apr 23 2017

Description: 
1. given arbitary number of processes (user-defined) to parallelly sort a large unsorted data set (10,000 elements)
2. define bins (which collects a certain range of integers), and allocate the elements into bins
3. collective communication in mpi4py is used for assigning tasks and gather the result
4. the order of sorted array is checked for correctness

Assumptions:
1. to simplify the problem, only integers are considered
2. the number of processes is defined by users when calling the program in cmd tool, and the length of data set is set to be 10,000

Call by: mpiexec -n <the total number of processes> python parallel_sorter.py 
'''

#parallel with collective communication 

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size >= 1

array_size = 10000

data = np.array([])
data_sorted = np.array([])

# step 1: generate a large unsorted data set of size 10,000 and slice the array into bins by value
if rank == 0 :
	# root process: generate a large unsorted data set
	data = np.random.randint(low=0, high=array_size, size=array_size)
	print(data)

	# define the amount of bins and their size
	bin_num = size
	bin_size = array_size // bin_num
	if array_size % bin_num != 0:
		bin_size = bin_size + 1

	# create bins and slice the array into bins by value
	data_sent = list()
	for i in range(size):
		data_sent.append([])
	for i in range(array_size):
		element = data[i]
		bin_idx = element // bin_size
		data_sent[bin_idx] = np.append(data_sent[bin_idx], element)

else:
	data_sent = None

# step 2 : send each bin to the process by collective communication methed scatter()
data_process = comm.scatter(data_sent, root=0)

# step 3 every process sorts their own tasks
data_process = sorted(data_process)

# step 4 : gather the sorted result back to process 0 by collective communication methed gather()
data_sorted = comm.gather(data_process, root=0)

# step 5: verification the increasing order of the sorted array
if rank == 0:
	data_sorted = np.concatenate(data_sorted)
	print(data_sorted)

if rank == 0:
	# check the order
	sort_b = True
	for i in range(array_size-1):
		try:
			sort_b = data_sorted[i] <= data_sorted[i+1]
		except TypeError:
			continue

	if sort_b == True:
		print("SORTED")
	else:
		print("UNSORTED")


