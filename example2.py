import simulations
import visualisations
import numpy as np
from functools import partial

# parameters
# maximum accelerations
A = np.array([0.5,0.5])
# optimal velocities
opt = np.array([6,10])
# kappa
kap = 10
# K_j
K=np.array([1,1])
# omega
om = 10
# initial distances and velocities
initval = np.array([5, 4, 1, 1])

d = partial(simulations.derivative, A=A, optimal_velocity=opt, kappa=kap, K=K, omega=om)
t, v = simulations.euler(initial_time=0, end_time=20, number_of_points=1000000,
                         initial_value=initval, derivative=d)

visualisations.visualise(t, v, filename='plot1')