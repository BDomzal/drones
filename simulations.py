import numpy as np
from functools import partial


def euler(initial_time, end_time, number_of_points, initial_value, derivative):
    """
    Estimates the solution of ODE using Euler's method.
    :param initial_time: initial time.
    :param end_time: end time.
    :param number_of_points: how many points should be between initial time and end time.
    :param initial_value: value of solution of ODE in initial time.
    :param derivative: function (R^(n+1)->R^n) representing right hand side of ODE: x'(t)=f(t,x). It should take two
    arguments: t and x, where t is float and x is a numpy array with shape (n,). Values of a function should be numpy
    arrays with shape (n,).
    :return: numpy array representing time vector and numpy array representing estimated values of solution of ODE.
    """
    time = np.linspace(start=initial_time, stop=end_time, num=number_of_points)
    values = np.empty((time.shape[0], initial_value.shape[0]))
    intervals_lengths = time[1:] - time[:-1]
    values[0, :] = initial_value
    for row in range(1, values.shape[0]):
        values[row, :] = values[row-1, :] + (derivative(time[row-1], values[row-1, :])*intervals_lengths[row-1])
    return time, values


def is_in_front(i, distance):
    """
    Checks if object i is in front of other objects.
    :param i: index of object of interest.
    :param distance: numpy 1D array representing locations/distance covered by objects.
    :return: numpy 1D binary array with the same length as distance. 1 on j-th position indicates that i-th object
    is in front of j-th object.
    """
    return (distance < distance[i])*1


def model1(A, velocity, optimal_velocity, kappa, K, distance, omega):
    """
    Function calculating value of second derivative according to first version of our model. n - number of objects.
    :param A: maximal accelerations, shape (n,)
    :param velocity: current velocity of objects, shape (n,)
    :param optimal_velocity: optimal velocity of objects, shape (n,)
    :param kappa: float
    :param K: shape (n,)
    :param distance: current location of objects/distance covered, shape(n,)
    :param omega: float
    :return: np.array with shape (n,) representing accelerations of objects.
    """
    dim = distance.shape[0]
    exp_vector = np.exp((-1)*distance/omega)
    bool_array = np.empty((dim, dim))
    for i in range(dim):
        bool_array[:, i] = K[i]*is_in_front(i, distance)
    result = A*(1+(-1)*velocity/optimal_velocity-(1/kappa)*np.exp((1/omega)*distance)*np.matmul(bool_array, exp_vector))
    return result


def model2(A, velocity, optimal_velocity, kappa, K, distance, omega):
    """
    Function calculating value of second derivative according to second version of our model. n - number of objects.
    :param A: maximal accelerations, shape (n,)
    :param velocity: current velocity of objects, shape (n,)
    :param optimal_velocity: optimal velocity of objects, shape (n,)
    :param kappa: float
    :param K: shape (n,)
    :param distance: current location of objects/distance covered, shape(n,)
    :param omega: float
    :return: np.array with shape (n,) representing accelerations of objects.
    """
    dim = distance.shape[0]
    exp_vector = np.exp((-1)*distance/omega)
    bool_array = np.empty((dim, dim))
    for i in range(dim):
        bool_array[:, i] = K[i]*is_in_front(i, distance)
    result = A*(1+(-1)*velocity/optimal_velocity-(1/kappa)*velocity*np.exp((1/omega)*distance)*np.matmul(bool_array, exp_vector))
    return result


def derivative(t, x, A, optimal_velocity, kappa, K, omega, model):
    """
    Function calculating derivative of y=(x,x') according to our model. n - number of objects.
    :param t: time (not used in our model).
    :param x: np.array with shape (2*n,). First n coordinates correspond to location/distance covered by objects from
    0-th to (n-1)-th. Coordinates from (n+1)-th to 2n-th correspond to velocities of objects from 0-th to (n-1)-th.
    :param A: see: model.
    :param optimal_velocity: see: model.
    :param kappa: see: model.
    :param K: see: model.
    :param omega: see: model.
    :param model: which model should be used.
    :return: np.array with shape (2*n,). First n coordinates correspond to velocities of objects from 0-th to (n-1)-th.
    Coordinates from (n+1)-th to (2*n)-th correspond to accelerations of objects from 0-th to (n-1)-th.
    """
    result = np.empty(x.size)
    n = int(x.size/2)
    result[:n, ] = x[n:, ]
    result[n:, ] = model(A=A, velocity=x[n:, ], optimal_velocity=optimal_velocity, kappa=kappa, K=K, distance=x[:n, ],
                         omega=omega)
    return result
