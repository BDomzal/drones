import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt, animation, patches
import matplotlib.pyplot as plt
from textwrap import wrap


def visualise(time, values, drone_radius=None, drone_height=None, heightmax=None, heightmin=None, filename='plot'):
    """
    Function produces animated simulation of the drones movement. Based on the results from simulations functions.
    :param time: vector of timestamps
    :param values: positions of drones at the timestamps
    :return:
    """
    frame_num, drone_num = values.shape
    drone_num //= 2
    drone_col = sns.color_palette(None, drone_num)
    end_distance = np.max(values[-1,])+3

    # TODO: should we draw heights from uniform or from normal distribution?
    if drone_height is None:
        drone_height = [random.uniform(1, 20) for i in range(drone_num)]
        heightmax = 25
        heightmin = -5
    # TODO: drones size as parameter, connected with wind force
    # changed drone shape to rectangle
    if drone_radius is None:
        drone_radius = [1 for i in range(drone_num)]
    drone_width = 0.01*end_distance

    fig, ax = plt.subplots(figsize=(10, 5))

    def anim_frame(k):
        plt.cla()
        ax.set_xlim((0, end_distance))
        ax.set_ylim((heightmin, heightmax))
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Polozenie')
        for i in range(drone_num):
            # # circle
            # ax.add_patch(plt.Circle((values[k, i], drone_height[i]), drone_radius[i], color=drone_col[i]))
            # rectangle
            ax.add_patch(patches.Rectangle(xy=(values[k, i], drone_height[i]), height=drone_radius[i], width=drone_width, color=drone_col[i]))
        return ax

    anim = animation.FuncAnimation(fig, func=anim_frame, frames=np.arange(0, frame_num, 1000), interval=0.015)

    Writer = animation.writers['ffmpeg']
    writer1 = Writer(fps=60, metadata={'artist': 'Me'}, bitrate=1800)
    anim.save(filename+'.mp4', writer=writer1)

    # If we need to save it as gif
    # anim.save('pplot1.gif', writer='imagemagick')

def visualise_wind(time, values, wind, timeswitch, drone_radius=None, drone_height=None, heightmax=None, heightmin=None, filename='plot'):
    """
    Function produces animated simulation of the drones movement. Based on the results from simulations functions.
    :param time: vector of timestamps
    :param values: positions of drones at the timestamps
    :return:
    """
    frame_num, drone_num = values.shape
    drone_num //= 2
    drone_col = sns.color_palette(None, drone_num)
    end_distance = np.max(values[-1,])+3

    if drone_height is None:
        drone_height = [random.uniform(1, 20) for i in range(drone_num)]
        heightmax = 25
        heightmin = -5
    if drone_radius is None:
        drone_radius = [1 for i in range(drone_num)]
    drone_width = 0.01*end_distance

    fig, ax = plt.subplots(figsize=(10, 5))

    def anim_frame(k):
        plt.cla()
        ax.set_xlim((0, end_distance))
        ax.set_ylim((heightmin, heightmax))
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Polozenie')
        if time[k]<timeswitch:
            ax.set_title('H='+str(wind[0]))
        else:
            ax.set_title('H=' + str(wind[1]))
        for i in range(drone_num):
            # # circle
            # ax.add_patch(plt.Circle((values[k, i], drone_height[i]), drone_radius[i], color=drone_col[i]))
            # rectangle
            ax.add_patch(patches.Rectangle(xy=(values[k, i], drone_height[i]), height=drone_radius[i], width=drone_width, color=drone_col[i]))
        return ax

    anim = animation.FuncAnimation(fig, func=anim_frame, frames=np.arange(0, frame_num, 1000), interval=0.015)

    Writer = animation.writers['ffmpeg']
    writer1 = Writer(fps=60, metadata={'artist': 'Me'}, bitrate=1800)
    anim.save(filename+'.mp4', writer=writer1)

    # If we need to save it as gif
    # anim.save('pplot1.gif', writer='imagemagick')

def visualise_position_against_time(t, v, results_path=None, title='simulation.png', show=True, labels=None):
    """
    Plot the position of each drone against time for the simulation output (original model format).

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array of shape (len(t), 2*n) containing positions; columns 0..n-1 are positions used here
    :param results_path: optional path where the figure will be saved (filename appended to this path)
    :param title: filename to use when saving the figure (default 'simulation.png')
    :param show: whether to call plt.show() after plotting
    :param labels: optional list of legend labels; if None, defaults to 'drone 0', 'drone 1', ...
    :return: None (saves or shows the matplotlib figure)
    """

    n = v.shape[1]//2
    for i in range(n):
        plt.plot(t, v[:,i], label = 'drone ' + str(i) if labels is None else labels[i])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("time",fontsize=20)
    plt.ylabel("position",fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_gap(t, v, opt, kap, om, K, results_path=None, title='st_st_2.png', show=True):
    """
    Visualise the gap between drone 0 and drone 1 over time and compare it to a theoretical gap.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array with positions; expects columns where v[:,0] and v[:,1] are relevant
    :param opt: array-like of optimal velocities (used in theoretical calculation)
    :param kap: scalar parameter (kappa) for theoretical formula
    :param om: scalar parameter (omega) for theoretical formula
    :param K: array-like constant(s) used in theoretical formula
    :param results_path: optional path where the figure will be saved (filename appended to this path)
    :param title: filename to use when saving the figure (default 'st_st_2.png')
    :param show: whether to call plt.show() after plotting
    :return: None
    """

    fig, ax = plt.subplots()

    ax.plot(t,v[:,0]-v[:,1], label='\n'.join(wrap('gap between drone 0 and drone 1',20)))
    plt.plot(t,om*np.log((opt[0]/kap)*K[0]*np.exp((v[:,0]-v[:,0])/om)/(1-opt[0]/opt[1])), color='deeppink', label='theoretically calculated gap')

    leg = ax.legend(fontsize=20)
    plt.xlabel('time',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_difference(t, v, opt, kap, om, K, results_path=None, title='difference.png', show=True):
    """
    Plot the difference between the theoretically computed gap and the observed gap (v[:,0]-v[:,1]).

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array of positions
    :param opt: array-like of optimal velocities
    :param kap: scalar parameter kappa used in theoretical formula
    :param om: scalar parameter omega used in theoretical formula
    :param K: array-like constant(s) used in theoretical formula
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'difference.png')
    :param show: whether to call plt.show()
    :return: None
    """

    fig, ax = plt.subplots()

    plt.plot(t,om*np.log((opt[0]/kap)*K[0]*np.exp((v[:,0]-v[:,0])/om)/(1-opt[0]/opt[1]))-v[:,0]+v[:,1], color='fuchsia', label='\n'.join(wrap('difference',20)))

    leg = ax.legend(fontsize=15)
    plt.xlabel('time',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_uniform_state_with_gaps_for_5_drones(t, v, opt=np.array([1,2,2,2,2]), kap=10, om=10, K=np.array([50,50,50,50,50]), include_gap_length=True, results_path=None, title='st_st_1_plus_gap.png', show=True):
    """
    Plot positions for exactly 5 drones and optionally draw the theoretically calculated gap markers.

    This function expects the input `v` in the original model shape (columns for positions and perhaps velocities),
    and asserts that there are exactly 5 drones.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array with at least 5 position columns (shape (len(t), 2*n) expected by other code)
    :param opt: array-like of optimal velocities (default array for 5 drones)
    :param kap: scalar kappa parameter used in theoretical formula
    :param om: scalar omega parameter used in theoretical formula
    :param K: array-like constants used in theoretical formula
    :param include_gap_length: whether to draw vertical/horizontal lines visualising theoretical gap
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'st_st_1_plus_gap.png')
    :param show: whether to call plt.show()
    :return: None
    """

    n = v.shape[1]//2
    assert n==5

    for i in range(5):
        plt.plot(t, v[:,i], label = 'drone ' + str(i))

        # if i in [0,1]:
        #     plt.plot(t, v[:,i], label = 'drone ' + str(i))
        # else:
        #     plt.plot(t, v[:,i])

    if include_gap_length:
        gap = om*np.log((opt[0]/kap)*K[0]*np.exp((v[:,0]-v[:,0])/om)/(1-opt[0]/opt[1]))[0]
        plt.vlines(x=150, ymin=v[750000,1], ymax=v[750000,1]+gap, color="deeppink", label='\n'.join(wrap('theoretically calculated length of gap', 15)))
        plt.hlines(y=v[750000,1]+gap, xmin=150-3, xmax=150+3, color="deeppink", linewidth=3) 
        plt.hlines(y=v[750000,1], xmin=150-3, xmax=150+3, color="deeppink", linewidth=3) 
        
        plt.vlines(x=166, ymin=v[830000,2], ymax=v[830000,2]+gap, color="deeppink")
        plt.hlines(y=v[830000,2]+gap, xmin=166-3, xmax=166+3, color="deeppink", linewidth=3) 
        plt.hlines(y=v[830000,2], xmin=166-3, xmax=166+3, color="deeppink", linewidth=3) 

        plt.vlines(x=182, ymin=v[910000,3], ymax=v[910000,3]+gap, color="deeppink")
        plt.hlines(y=v[910000,3]+gap, xmin=182-3, xmax=182+3, color="deeppink", linewidth=3) 
        plt.hlines(y=v[910000,3], xmin=182-3, xmax=182+3, color="deeppink", linewidth=3) 

        plt.vlines(x=196, ymin=v[990000,4], ymax=v[990000,4]+gap, color="deeppink")
        plt.hlines(y=v[990000,4]+gap, xmin=196-3, xmax=196+3, color="deeppink", linewidth=3) 
        plt.hlines(y=v[990000,4], xmin=196-3, xmax=196+3, color="deeppink", linewidth=3) 
        
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("time",fontsize=20)
    plt.ylabel("position",fontsize=20)
    plt.legend(fontsize=17)
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_macro(t, v, results_path=None, title='macro4.png', show=True):
    """
    Plot positions against time for 5 obstacles and 30 drones.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array where columns 0..4 correspond to obstacles and 5..35 correspond to drones
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'macro4.png')
    :param show: whether to call plt.show()
    :return: None
    """

    # obstacles
    plt.plot(t, v[:,0], color='orange', label='obstacles')
    for i in range(1, 5):
        plt.plot(t, v[:,i], color='orange')
        
    # drones
    plt.plot(t, v[:,5], color='grey', label='drones')
    for i in range(6, 36):
        plt.plot(t, v[:,i], color='grey')
        
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.xlabel("time", fontsize=20)
    plt.ylabel("position", fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_positive_potential_no_passing(t, v, initial_velocity = np.array([3, 0]), A=np.array([1, 2]), opt=np.array([3, 4]), kap=1, om=4, K=np.array([0.6, 1]), include_gap_length=True, results_path=None, title='positive_potential_no_passing.png', show=True):
    """
    Visualise a two-drone scenario with positive potential where passing does not occur.

    The function plots both drone trajectories and draws the theoretically calculated gap length
    (based on provided parameters) as vertical/horizontal markers.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array with at least two position columns (v[:,0], v[:,1])
    :param initial_velocity: initial velocity vector used to compute the gap (default [3, 0])
    :param A: parameter array A used by the model (unused in plotting but kept for API consistency)
    :param opt: optimal velocities array
    :param kap: scalar kappa parameter
    :param om: scalar omega parameter
    :param K: array-like constants used in theoretical calculation
    :param include_gap_length: whether to draw the theoretical gap markers
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'positive_potential_no_passing.png')
    :param show: whether to call plt.show()
    :return: None
    """

    gap = om*np.log((opt[0]/kap)*K[0]*np.exp((initial_velocity[0]-initial_velocity[0])/om)/(1-opt[0]/opt[1]))

    for i in range(2):
        plt.plot(t, v[:,i], label = 'drone ' + str(i))

    plt.vlines(x=7.5, ymin=v[500000,1], ymax=v[500000,1]+gap, color="deeppink", label='\n'.join(wrap('theoretically calculated length of gap', 27)))
    plt.hlines(y=v[500000,1]+gap, xmin=7.5-0.15, xmax=7.5+0.15, color="deeppink", linewidth=3) 
    plt.hlines(y=v[500000,1], xmin=7.5-0.15, xmax=7.5+0.15, color="deeppink", linewidth=3) 

    plt.xlim(0,15)
    plt.ylim(0,50)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("time",fontsize=20)
    plt.ylabel("position",fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_covered_distance(t, v, opt, results_path=None, title='max_vel.png', show=True):
    """
    Plot covered distance over time vs. a reference maximal-velocity straight-line distance.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array where v[:,0] contains the covered distance to plot
    :param opt: array-like where opt[0] is used as the maximal velocity reference
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'max_vel.png')
    :param show: whether to call plt.show()
    :return: None
    """

    fig, ax = plt.subplots()

    ax.plot(t, v[:,0], label='\n'.join(wrap('distance covered during flight with the help of wind', 20)))
    ax.plot(t, v[0,0]+opt[0]*t, color='deeppink', label='\n'.join(wrap('distance covered during flight with maximal velocity',20)))

    leg = ax.legend(fontsize=20);
    plt.xlabel('time',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_single_drone_and_changing_wind(t, v, results_path=None, title='neg_vel3.png', show=True, text1 = '$H_0(t)=0$', text2 = '$H_0(t)=5$', text3 = '$H_0(t)=-10$'):
    """
    Plot a single drone trajectory and annotate the plot with textual descriptions of wind phases.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array where v[:,0] is the trajectory of the single drone
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'neg_vel3.png')
    :param show: whether to call plt.show()
    :param text1: annotation text for the first wind phase (rendered in a text box)
    :param text2: annotation text for the second wind phase
    :param text3: annotation text for the third wind phase
    :return: None
    """

    plt.plot(t, v[:,0], label = 'drone ' + str(0))
    plt.xlabel("time",fontsize=20)
    plt.ylabel("position",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)


    props = dict(boxstyle='round', facecolor='aliceblue', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(0, 7.2, text1, fontsize=20,
        verticalalignment='top', bbox=props, rotation=27)
    plt.text(5, 17.5, text2, fontsize=20,
        verticalalignment='top', bbox=props, rotation=61)
    plt.text(11, 14, text3, fontsize=20,
        verticalalignment='top', bbox=props, rotation=-63)

    plt.tight_layout()
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_position_against_time_SCM(t, v, results_path=None, title='simulation.png', show=True, labels=None):
    """
    Plot positions against time for the Scalar Capacity Model (SCM) format output.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array of shape (len(t), n) containing positions for n drones
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'simulation.png')
    :param show: whether to call plt.show()
    :param labels: optional list of legend labels; if None, defaults to 'drone 0', 'drone 1', ...
    :return: None
    """

    n = v.shape[1]
    for i in range(n):
        plt.plot(t, v[:,i], label = 'drone ' + str(i) if labels is None else labels[i])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("time",fontsize=20)
    plt.ylabel("position",fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()

def visualise_macro_scm(t, v, time_limit = 50000, results_path=None, title='SCM_macro4.png', show=True):
    """
    Macro plot for SCM-format data showing obstacles and drones up to a time index limit.

    :param t: 1D array-like of time stamps
    :param v: 2D numpy array where column 0..4 are obstacles and 5..35 are drones (SCM layout)
    :param time_limit: integer index limit to slice the time and v arrays for plotting (default 50000)
    :param results_path: optional path where the figure will be saved
    :param title: filename for saving (default 'SCM_macro4.png')
    :param show: whether to call plt.show()
    :return: None
    """

    # obstacles
    plt.plot(t[:time_limit], v[:time_limit,0], color='orange', label='obstacles')
    for i in range(1, 5):
        plt.plot(t[:time_limit], v[:time_limit,i], color='orange')
        
    # drones
    plt.plot(t[:time_limit], v[:time_limit,5], color='grey', label='drones')
    for i in range(6, 36):
        plt.plot(t[:time_limit], v[:time_limit,i], color='grey')
        
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.xlabel("time", fontsize=20)
    plt.ylabel("position", fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    if results_path is not None:
        plt.savefig(results_path + title, dpi=400)
    if show:
        plt.show()