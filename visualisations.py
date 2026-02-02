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