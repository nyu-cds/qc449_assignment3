'''
Author: Qiming Chen qc449@nyu.edu
Date: May 7 2017
Description: 1. Calculate the product of integer 1-1000 by creating a RDD with parallelize() and aggregate the values by fold(). 
			 2. Verify the result by math.factorial()
'''

from pyspark import SparkContext
from operator import mul
import math

if __name__ == '__main__':
	# configuration
	sc = SparkContext("local", "product")

	# Create an RDD of numbers from 1 to 1,000
	nums = sc.parallelize(range(1,1001))

	# Compute the product of all values in the RDD
	p = nums.fold(1, mul)
	print("Product of 1~1000 is: ", p)

	# Verification with math.factorial()
	p2 = math.factorial(1000)
	print("The spark result matches with the one with function math.factorial(1000): ", True if (p2-p)==0 else False)