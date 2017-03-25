"""
Author: Qiming Chen qc449@nyu.edu
Comment: 

N-body simulation.

    Version: Optimized by Cython, derived from nbody_opt.py
    
    Speed up: R = 95.9 / 29.6 = 3.24

    1. initial: 33.6s
    2. adding %%cython in jupyter notebook 33.6s -> 13.6s
    3. adding cdef for global variable 13.6s -> 13s
    4. claiming C type variable for function parameters 13s -> 13.4s
    5. adding cdef for local variables 13.4s -> 7.23s

    tested in jupyter notebook by 

        import pyximport
        pyximport.install()
        import nbody_cython
        %timeit nbody_cython.nbody(100, 'sun', 20000)

    1 loop, best of 3: 6.98 s per loop

    Speed up: R = 33.6 / 6.98 = 4.81

"""

import cython
from itertools import combinations
import numpy as np

cdef double PI = 3.14159265358979323
cdef double SOLAR_MASS = 4 * PI * PI
cdef double DAYS_PER_YEAR = 365.24

cdef dict BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS), # SOLAR_MASS = 39.4784176044

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                5.15138902046611451e-05 * SOLAR_MASS)}

cdef advance(int iterations, set body_keypairs, dict bodies, double dt = 0.01):
    '''
        advance the system one timestep
    '''
    
#     cdef np.ndarray[np.double_t,ndim=1] l1, l2
    
    cdef list v1, v2, r
    cdef double x1, x2, y1, y2, z1, z2, m1, m2, mag, m1_mag, m2_mag, dx, dy, dz, m, vx, vy, vz
    bodies_keys = bodies.keys()

#     bodies_keys = bodies.keys()
    
    for _ in range(iterations):
        for (body1, body2) in iter(body_keypairs):         
            ([x1, y1, z1], v1, m1) = bodies[body1]
            ([x2, y2, z2], v2, m2) = bodies[body2]
            (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
            # update v's
            mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            m1_mag = m1 * mag
            m2_mag = m2 * mag
            v1[0] -= dx * m2_mag
            v1[1] -= dy * m2_mag
            v1[2] -= dz * m2_mag
            v2[0] += dx * m1_mag
            v2[1] += dy * m1_mag
            v2[2] += dz * m1_mag
            
        # for body in BODIES_KEYS:
        for i, body in enumerate(iter(bodies_keys)):
            (r, [vx, vy, vz], m) = bodies[body]
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz
            
cdef report_energy(dict bodies, set body_keypairs, double e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    
    cdef list v1, v2, r
    cdef double x1, x2, y1, y2, z1, z2, m1, m2, dx, dy, dz, vx, vy, vz, m
    bodies_keys = bodies.keys()
    
    for (body1, body2) in body_keypairs:
        ((x1, y1, z1), v1, m1) = bodies[body1]
        ((x2, y2, z2), v2, m2) = bodies[body2]

        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
        # comput energy
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)

    # for body in BODIES_KEYS:
    for i, body in enumerate(iter(bodies_keys)):
        (r, [vx, vy, vz], m) = bodies[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

cdef offset_momentum(dict bodies, str reference, double px=0.0, double py=0.0, double pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    
    cdef list r, v
    cdef double vx, vy, vz, m
    bodies_keys = bodies.keys()
    ref = bodies[reference]

    # for body in BODIES_KEYS:
    for i, body in enumerate(iter(bodies_keys)):
        (r, [vx, vy, vz], m) = bodies[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def nbody(int loops, str reference, int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    # Set up global state
    offset_momentum(BODIES, reference)

    cdef set body_pairs = set(combinations(BODIES,2)) # create unique pairs

    for _ in range(loops):
        advance(iterations, body_pairs, BODIES)
#         print(report_energy(BODIES, BODIES_KEYS, body_pairs))
        report_energy(BODIES, body_pairs)