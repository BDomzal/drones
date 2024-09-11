import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt, animation, patches


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




