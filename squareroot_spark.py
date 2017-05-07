'''
Author: Qiming Chen qc449@nyu.edu
Date: May 7 2017
Description: 1. A Spark application with pyspark to calculate the average of the square root of all the numbers from 1 to 1000. 
			 2. Verify the spark result by recalculating with numpy package
'''

# calculate the average of the square root of all the numbers from 1 to 1000. 
# i.e the sum of the square roots of all the numbers divided by 1000.
from pyspark import SparkContext
from operator import add
import numpy as np

if __name__ == '__main__':
	# Spark
	# Configuration
	sc = SparkContext("local", "sqrt_spark")

	# Create a RDD of integer 1~1000
	nums = sc.parallelize(range(1, 1001))

	# Sum up all elements in that RDD
	sum_sqr = nums.map(lambda x: x ** 0.5).fold(0, add)

	# Average over the size of the RDD
	result = sum_sqr / (nums.count())
	print("The square_root is: ", result)

	# Verification
	nums2 = np.arange(1, 1001, dtype='f')
	nums2_sqr = np.sqrt(nums2)
	result2 = np.sum(nums2_sqr) / (len(nums2_sqr))
	print("The spark result matches with a sequential result with numpy packages: ",True if abs(result2 - result)<0.0001 else False)
