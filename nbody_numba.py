"""
    N-body simulation.

    Optimized by numba

    Add @jit decorators to all funcitons
    Add function signatures to all funcitons
    Add a vectorized ufunc to the nbody_numba.py program called vec_deltas

    Change structure of BODIES
"""
import numpy as np
from numba import void, jit, int32, float64, typeof, char, vectorize, int_
from itertools import combinations

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES_KEYS = np.array( ['sun', 'jupiter', 'saturn', 'uranus', 'neptune'])

BODIES = np.array( [ ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [39.4784176044, 0.0, 0.0]),
                  
                ([4.84143144246472090e+00,-1.16032004402742839e+00,-1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,7.69901118419740425e-03 * DAYS_PER_YEAR, -6.90460016972063023e-05 * DAYS_PER_YEAR],
                [9.54791938424326609e-04 * 39.4784176044, 0.0, 0.0] ),

                ([8.34336671824457987e+00,4.12479856412430479e+00,-4.03523417114321381e-01],
                [-2.76742510726862411e-03 * DAYS_PER_YEAR,4.99852801234917238e-03 * DAYS_PER_YEAR,2.30417297573763929e-05 * DAYS_PER_YEAR],
                [2.85885980666130812e-04 * 39.4784176044, 0.0, 0.0] ),
                
                ([1.28943695621391310e+01,-1.51111514016986312e+01,-2.23307578892655734e-01],
                 [2.96460137564761618e-03 * DAYS_PER_YEAR,2.37847173959480950e-03 * DAYS_PER_YEAR,-2.96589568540237556e-05 * DAYS_PER_YEAR],
                 [4.36624404335156298e-05 * 39.4784176044, 0.0, 0.0] ),
                  
                ([1.53796971148509165e+01,-2.59193146099879641e+01,1.79258772950371181e-01],
                 [2.68067772490389322e-03 * DAYS_PER_YEAR,1.62824170038242295e-03 * DAYS_PER_YEAR,-9.51592254519715870e-05 * DAYS_PER_YEAR],
                 [5.15138902046611451e-05 * 39.4784176044, 0.0, 0.0] ) ],
                  
                  dtype = np.float64
                 )


@vectorize([float64(float64, float64)])
def vec_deltas(x, y):
    return x - y

@jit(int32(char))
def get_index(reference):
    idx = 0;
    while(BODIES_KEYS[idx] != reference ):
        idx += 1
    return idx;
    

@jit(void(int32, int32[:,:], float64))
def advance(iterations, body_keypairs, dt):
    '''
        advance the system one timestep
    '''
    length = len(BODIES_KEYS)
    for _ in range(iterations):
        for (body1, body2) in body_keypairs:
            (l1, v1, m1) = BODIES[body1]
            (l2, v2, m2) = BODIES[body2]
            (dx, dy, dz) = vec_deltas(l1,l2)
            # update v's
            mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            m1_mag = m1[0] * mag
            m2_mag = m2[0] * mag
            v1[0] -= dx * m2_mag
            v1[1] -= dy * m2_mag
            v1[2] -= dz * m2_mag
            v2[0] += dx * m1_mag
            v2[1] += dy * m1_mag
            v2[2] += dz * m1_mag
        
        for body in range(length):
            (r, [vx, vy, vz], m) = BODIES[body]
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz

@jit(void(int32[:,:],float64))
def report_energy(body_keypairs, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    length = len(BODIES_KEYS)
    
    for (body1, body2) in body_keypairs:
        ([x1,y1,z1], v1, m1) = BODIES[body1]
        ([x2,y2,z2], v2, m2) = BODIES[body2]
        (dx, dy, dz) = (x1-x2, y1-y2,z1-z2)
        # comput energy
        e -= (m1[0] * m2[0]) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    
    for body in range(length):
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m[0] * (vx * vx + vy * vy + vz * vz) / 2.
    
    return e

@jit(void(char, float64, float64, float64))
def offset_momentum(reference, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    
    length = len(BODIES_KEYS)
    idx = get_index(reference)
    ref = BODIES[idx]
    
    for body in range(length):
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m[0]
        py -= vy * m[0]
        pz -= vz * m[0]
        
    (r, v, m) = ref
    v[0] = px / m[0]
    v[1] = py / m[0]
    v[2] = pz / m[0]

@jit(void(int32,char,float64))
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    # Set up global state

    offset_momentum(reference)

    body_pairs = np.array(list(combinations(range(len(BODIES_KEYS)),2)), dtype = np.int32)
    
    for _ in range(loops):
        # report_energy(Bodies, body_pairs)
        advance(iterations, body_pairs, 0.01)
        print(report_energy(body_pairs))

if __name__ == '__main__':
    nbody(100, 'sun', 20000)

