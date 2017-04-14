''' 
Author: Qiming Chen
Date: Apr 12 2017
Description: print “Hello” with even rank and print “Goodbye” with odd rank.
Call by: mpiexec -n <the total number of processes> python mpi_assignment_1.py 
'''

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # index of a particular process

if rank % 2 == 0: # even
	print("Hello from process", rank)
else: # odd
	print("Goodbye from process", rank)
