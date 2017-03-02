# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 

"""
Qiming Chen qc449@nyu.edu

Original Overhead: 
    1000014 function calls in 1.898 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.969    0.484    0.972    0.486 <ipython-input-34-2fda8cf3c7a1>:16(multiply)
        1    0.463    0.463    0.464    0.464 <ipython-input-34-2fda8cf3c7a1>:3(add)
        1    0.378    0.378    0.459    0.459 <ipython-input-34-2fda8cf3c7a1>:29(sqrt)
  1000000    0.079    0.000    0.079    0.000 {built-in method math.sqrt}
        4    0.007    0.002    0.007    0.002 {built-in method numpy.core.multiarray.zeros}
        1    0.003    0.003    1.898    1.898 <string>:1(<module>)
        1    0.000    0.000    1.898    1.898 {built-in method builtins.exec}
        1    0.000    0.000    1.895    1.895 <ipython-input-34-2fda8cf3c7a1>:42(hypotenuse)

We can propose that there are two sources of the overhead:
    1. user-defined mutiply() add() sqrt() functions
    2. extra cost when calling these 3 functions (need to unwrap if possible)

Possible solutions:
    We will take advantage of some of the functions in Numpy package. 
    Since the operations defined by user is element-wise, so we can use
        numpy.add() to replace add()
        numpy.multiply() to replace multiply()
        numpy.sqrt() to replace sqrt()

Profiler: cProfiler and line_profiler, environment Jupyter Notebook with kernel python 3.5
    
    import cProfile
    %prun hypotenuse(A,B)

    %load_ext line_profiler
    %lprun -f hypotenuse hypotenuse(A,B)


Step 1: modify a function at one time to evaluate the improvement

(1) modify add() function
    replace the loop by np.add(x,y)

    1000014 function calls in 1.416 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.951    0.476    0.958    0.479 <ipython-input-19-3def0c702746>:16(multiply)
        1    0.376    0.376    0.449    0.449 <ipython-input-19-3def0c702746>:29(sqrt)
  1000000    0.073    0.000    0.073    0.000 {built-in method math.sqrt}
        1    0.005    0.005    0.005    0.005 <ipython-input-20-bcd22927d2f5>:1(add)
        1    0.000    0.000    1.412    1.412 <ipython-input-19-3def0c702746>:42(hypotenuse)

    The overhead of add() reduces from 0.458s to 0.005s

(2) modify multiply() function 
    replace the loop by np.multiply(x,y)

    1000013 function calls in 0.954 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.489    0.489    0.491    0.491 <ipython-input-28-3def0c702746>:3(add)
        1    0.379    0.379    0.454    0.454 <ipython-input-28-3def0c702746>:29(sqrt)
  1000000    0.075    0.000    0.075    0.000 {built-in method math.sqrt}
        2    0.006    0.003    0.006    0.003 <ipython-input-29-40c591ae665b>:1(multiply)
        1    0.000    0.000    0.952    0.952 <ipython-input-28-3def0c702746>:42(hypotenuse)
        
    The overhead of multiply() reduces from 0.958s to 0.006s

(3) modify sqrt() function 
    replace the loop by np.sqrt(x,y)

    12 function calls in 1.420 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.934    0.467    0.938    0.469 <ipython-input-9-b142caa8fdc3>:16(multiply)
        1    0.472    0.472    0.473    0.473 <ipython-input-9-b142caa8fdc3>:3(add)
        1    0.004    0.004    0.004    0.004 <ipython-input-12-8fdeadc38139>:1(sqrt)
        1    0.000    0.000    1.416    1.416 <ipython-input-9-b142caa8fdc3>:42(hypotenuse)

    The overhead of sqrt() reduces from 0.454s to 0.004s

Step 2: Modify all 3 functions
    
    8 function calls in 0.017 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.007    0.003    0.007    0.003 <ipython-input-32-4629f01ea78d>:1(multiply)
        1    0.004    0.004    0.004    0.004 <ipython-input-31-480cf63584c4>:1(add)
        1    0.004    0.004    0.004    0.004 <ipython-input-33-2382ba8da7d3>:1(sqrt)
        1    0.000    0.000    0.017    0.017 {built-in method builtins.exec}
        1    0.000    0.000    0.014    0.014 <ipython-input-28-3def0c702746>:42(hypotenuse)
        1    0.000    0.000    0.017    0.017 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    The overhead reduces from 1.898s to 0.017s

Step 3: Clean the code and Unwrap the functions
    
    4 function calls in 0.016 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.013    0.013    0.013    0.013 <ipython-input-11-5d2daf6f7af1>:1(hypotenuse)
        1    0.003    0.003    0.016    0.016 <string>:1(<module>)
        1    0.000    0.000    0.016    0.016 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    The overhead slightly reduces

###  Conclusion ###

## cProfile ##
................
Before 
1000014 function calls in 1.898 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.969    0.484    0.972    0.486 <ipython-input-34-2fda8cf3c7a1>:16(multiply)
        1    0.463    0.463    0.464    0.464 <ipython-input-34-2fda8cf3c7a1>:3(add)
        1    0.378    0.378    0.459    0.459 <ipython-input-34-2fda8cf3c7a1>:29(sqrt)
  1000000    0.079    0.000    0.079    0.000 {built-in method math.sqrt}
        4    0.007    0.002    0.007    0.002 {built-in method numpy.core.multiarray.zeros}
        1    0.003    0.003    1.898    1.898 <string>:1(<module>)
        1    0.000    0.000    1.898    1.898 {built-in method builtins.exec}
        1    0.000    0.000    1.895    1.895 <ipython-input-34-2fda8cf3c7a1>:42(hypotenuse)
        1    0.000    0.000    0.000    0.000 <frozen importlib._bootstrap>:996(_handle_fromlist)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
.................
After
4 function calls in 0.016 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.013    0.013    0.013    0.013 <ipython-input-30-93f7e97aea8c>:1(hypotenuse)
        1    0.003    0.003    0.016    0.016 <string>:1(<module>)
        1    0.000    0.000    0.016    0.016 {built-in method builtins.exec}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

Timer unit: 1e-06 s

.................
liner_profiler
Before
Timer unit: 1e-06 s

Total time: 3.42712 s
File: <ipython-input-34-2fda8cf3c7a1>
Function: hypotenuse at line 42

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    42                                           def hypotenuse(x,y):
    43         1       916636 916636.0     26.7      xx = multiply(x,x)
    44         1       875457 875457.0     25.5      yy = multiply(y,y)
    45         1       884458 884458.0     25.8      zz = add(xx, yy)
    46         1       750568 750568.0     21.9      return sqrt(zz)

...................
After
Total time: 0.013335 s
File: <ipython-input-30-93f7e97aea8c>
Function: hypotenuse at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           def hypotenuse(x,y):
     2         1         3082   3082.0     23.1      xx = np.multiply(x,x)
     3         1         2602   2602.0     19.5      yy = np.multiply(y,y)
     4         1         3737   3737.0     28.0      zz = np.add(xx, yy)
     5         1         3914   3914.0     29.4      return np.sqrt(zz)


COMMENT: using Numpy significantly improves the performance

    cProfiler: 1.895s -> 0.016s  speedup: 1.895/0.016 = 118.4375
    line_profiler 3.427s -> 0.013s speedup: 3.427/0.013 = 263.6154

"""

import numpy as np

def hypotenuse(x,y):
    xx = np.multiply(x,x)
    yy = np.multiply(y,y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)

if __name__ == '__main__':

    np.random.seed(1)

    M = 10**3
    N = 10**3

    A = np.random.random((M,N))
    B = np.random.random((M,N))

    # import cProfile
    # %prun hypotenuse(A,B)

    # %load_ext line_profiler
    # %lprun -f hypotenuse hypotenuse(A,B)
