'''
Author: Qiming Chen
Date: Apr 12 2017
Description: user defines the total number of processes and a starting integer less than 100, 
             process 0 multiply the number by 1 and send to process 1 after confirming the existence of process 1
             process i will multiply the number by (i+1) and send to process i+1 after confirming the existence of process i+1
Call by: mpiexec -n <the total number of processes> python mpi_assignment_2.py 
'''

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

size = comm.Get_size()
num = np.array([0])

if rank == 0:
	# Get starting number (integer, less than 100)
	while True:

		# Get starting number and verify if it is an integer
		start_num = 0
		try:
			input_str = input("Enter The starting number: ")
			start_num = int(input_str)
		except ValueError:
			print("Exception: Illegal input", input_str, "is not integer")
			continue

		# Verify the integer is less than 100
		if start_num >= 100:
			print("Exception: illegal starting number larger than or equal to 100. Try again.")
		else:
			break

	print("Process", rank, "reads a value", start_num, "from the user and verifies that it is an integer less than 100.")

	# multiply with next rank and send to next process after verifying the existence of next process
	# if true, send the number; otherwise, print exception
	num[0] = start_num * (1 + rank )
	if(size > 1 + rank):
		print("Process", rank, "sends", num[0], "to Process", rank + 1)
		comm.Isend(num, dest=1)
	else:
		print("Exception: no Process", 1, "exists")
        
if rank > 0:
	req = comm.Irecv(num, source=rank-1)
	req.Wait()
	print("Process", rank, "received the number", num[0])
	num[0] *= (1 + rank) 
	if(size > 1 + rank):
		print("Process", rank, "sends", num[0], "to Process", rank + 1)
		comm.Isend(num, dest=rank+1)
	else:
		print("Exception: no Process", rank+1, "exists")
		print("Program ends.")
        