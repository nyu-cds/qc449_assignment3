'''
Author: Qiming Chen qc449@nyu.edu
Date: May 7 2017
Description: a program to count the number of distinct words using pyspark, derived from the word count application with pyspark
Call by: spark-submit distinct_spark.py
'''

from pyspark import SparkContext
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':

	# configuration
	sc = SparkContext("local", "distinct_words")

	# flatten text to key-value pairs
	text = sc.textFile('pg2701.txt')
	words = text.flatMap(splitter)
	words_mapped = words.map(lambda x: (x, 1))

	# count the number of distinct words
	distinct_words = words_mapped.keys().distinct().count()
	print("The number of distinct words: ", distinct_words)
