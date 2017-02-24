'''

Description: taking two arguments n and k and prints all binary strings of length n that contain k zero bits, one per line.
Author: Qiming Chen
Date: Feb 23 2017

'''

import binary # import itself <binary.py>
from itertools import permutations

def zbits(n, k):
	result = set()

	# generate a concatenated string of length n with k 0s and (n-k) 1s and all permutations of this string of the same length
	for item in permutations(''.join(['0'*k, '1'*(n-k)]),n): 
		result.add(''.join(item))

	print(result)

	return result

if __name__ == '__main__':
	assert binary.zbits(4, 3) == {'0100', '0001', '0010', '1000'}
	assert binary.zbits(4, 1) == {'0111', '1011', '1101', '1110'}
	assert binary.zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}

