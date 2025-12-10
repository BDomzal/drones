import numpy as np


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


def model2(A, velocity, optimal_velocity, kappa, K, distance, omega):
    """
    Function calculating value of second derivative according to the basic version of our model (without wind). n - number of objects.
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


def model3(t, A, velocity, optimal_velocity, kappa, K, distance, omega, wind_function, m):
    """
    Function calculating value of second derivative according to the version of our model containing wind. n - number of objects.
    :param t: time.
    :param A: maximal accelerations, shape (n,)
    :param velocity: current velocity of objects, shape (n,)
    :param optimal_velocity: optimal velocity of objects, shape (n,)
    :param kappa: float
    :param K: shape (n,)
    :param distance: current location of objects/distance covered, shape(n,)
    :param omega: float
    :param wind_function: function to calculate wind force. It can either be a float if the wind affects
    each of the drones identically or a np.array with shape (n,). Note that this function must take only one
    argument (time) so if your function takes more arguments you need to wrap it, for example using functools.partial.
    :param m: masses of consecutive drones, shape (n,)
    :return: np.array with shape (n,) representing accelerations of objects.
    """
    w = wind_function(t)
    dim = distance.shape[0]
    exp_vector = np.exp((-1)*distance/omega)
    bool_array = np.empty((dim, dim))
    for i in range(dim):
        bool_array[:, i] = K[i]*is_in_front(i, distance)
    result = A*(1+(-1)*velocity/optimal_velocity-(1/kappa)*velocity*np.exp((1/omega)*distance)*np.matmul(bool_array, exp_vector))+w/m
    return result

def scalar_capacity_model(optimal_velocity, kappa, distance, omega):
    """
    Function calculating value of derivative according to Scalar Capacity Model (Gharibi et al., 2021). n - number of objects.
    :param optimal_velocity: optimal velocity of objects, shape (n,)
    :param kappa: float
    :param distance: current location of objects/distance covered, shape(n,)
    :param omega: float
    :return: np.array with shape (n,) representing velocities of objects.
    """
    dim = distance.shape[0]
    exp_vector = np.exp((-1)*distance/omega)
    bool_array = np.empty((dim, dim))
    for i in range(dim):
        bool_array[:, i] = is_in_front(i, distance)
    result = optimal_velocity*(1-(1/kappa)*np.exp((1/omega)*distance)*np.matmul(bool_array, exp_vector))
    return result


def derivative(t, x, A, optimal_velocity, kappa, K, omega, model):
    """
    Function calculating derivative of y=(x,x') according to our model. n - number of objects.
    :param t: time.
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


def derivative_with_wind(t, x, A, optimal_velocity, kappa, K, omega, model, wind_function, m):
    """
    Function calculating derivative of y=(x,x') according to our model with wind. n - number of objects.
    :param t: time.
    :param x: np.array with shape (2*n,). First n coordinates correspond to location/distance covered by objects from
    0-th to (n-1)-th. Coordinates from (n+1)-th to 2n-th correspond to velocities of objects from 0-th to (n-1)-th.
    :param A: see: model.
    :param optimal_velocity: see: model.
    :param kappa: see: model.
    :param K: see: model.
    :param omega: see: model.
    :param model: which model should be used.
    :param wind_function: wind_function: function to calculate wind force. It can either be a float if the wind affects
    each of the drones identically or a np.array with shape (n,). Note that this function must take only one
    argument (time) so if your function takes more arguments you need to wrap it, for example using functools.partial.
    :param m: see: model.
    :return:
    """
    result = np.empty(x.size)
    n = int(x.size/2)
    result[:n, ] = x[n:, ]
    result[n:, ] = model(t=t, A=A, velocity=x[n:, ], optimal_velocity=optimal_velocity, kappa=kappa, K=K, distance=x[:n, ],
                             omega=omega, wind_function=wind_function, m=m)
    return result


def derivative_scalar_capacity(t, x, optimal_velocity, kappa, omega):
    """
    Function calculating derivative of x according to Scalar Capacity Model. n - number of objects.
    :param t: time.
    :param x: np.array with shape (n,). The n coordinates correspond to location/distance covered by objects from
    0-th to (n-1)-th.
    :param optimal_velocity: see: scalar_capacity_model.
    :param kappa: see: scalar_capacity_model.
    :param omega: see: scalar_capacity_model.
    :return: np.array with shape (n,). The n coordinates correspond to velocities of objects from 0-th to (n-1)-th.
    """
    result = scalar_capacity_model(optimal_velocity=optimal_velocity, kappa=kappa, distance=x, omega=omega)
    return result


def discontinuous_wind(t, change_time, wind_force):
    """
    Function calculating wind force. It assumes that wind affects each of the drone identically and that it's force
    changes from one value to another.
    :param t: time.
    :param change_time: a list of moments of time when wind force changes.
    :param wind_force: list of values of wind force in the consecutive time intervals. It must have one more element than change_time list.
    after change_time.
    :return: float number which is a wind force in the given moment of time.
    """
    assert len(change_time) + 1 == len(wind_force), 'Wind_force argument must be a list with one element more than change_time argument.'
    for i, change_t in enumerate(change_time):
        if t < change_t:
            return wind_force[i]
        else:
            pass
    return wind_force[-1]


def constant_wind(t, wind_force):
    """
    Function calculating wind force. It assumes that wind affects each of the drone identically and that it's constant.
    :param t: time.
    :param wind_force: float.
    :return:
    """
    return wind_force


def battista_wind(t, rho, C, Af, vw, direction):
    """
    Function calculatind wind force according to paper by Anthony Battista and Daiheng Ni [25].
    :param t: time; float.
    :param rho: mass density of air; float.
    :param C: drag coefficient of i-th drone; np.array with shape (n,).
    :param Af: frontal area of i-th drone; np.array with shape (n,).
    :param vw: wind velocity; float.
    :param direction: either 1 or -1
    :return: np.array with shape (n,) where i-th coordinate corresponds to wind force acting on i-th drone.
    """
    assert direction == 1 or direction == -1, "Direction should equal to 1 or -1."
    return direction*0.5*rho*(vw**2)*C*Af

