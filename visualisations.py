import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt, patches, animation


def visualise(time, values, radius=None, filename='plot'):
    """
    Function produces animated simulation of the drones movement. Based on the results from simulations functions.
    :param time: vector of timestamps
    :param values: positions of drones at the timestamps
    :return:
    """
    frame_num, drone_num = values.shape
    drone_num //= 2
    drone_col = sns.color_palette(None, drone_num)
    # TODO: czy wysokosci powinny byc losowane z rozkladu jednostajnego czy normalnego
    drone_height = [random.uniform(1, 20) for i in range(drone_num)]
    # TODO: wielkosc drona jako paramter, polaczone z funkcja wiatru
    # drone_rad = radius


    fig, ax = plt.subplots()

    def anim_frame(k):
        plt.cla()
        ax.set_xlim((0, np.max(values[-1,])))
        ax.set_ylim((-5, 25))
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Czas')
        for i in range(drone_num):
            ax.add_patch(plt.Circle((values[k, i], drone_height[i]), 1, color=drone_col[i]))
        return ax

    anim = animation.FuncAnimation(fig, func=anim_frame, frames=np.arange(0,frame_num,1000), interval=0.02)

    Writer = animation.writers['ffmpeg']
    writer1 = Writer(fps=60, metadata={'artist': 'Me'}, bitrate=1800)
    anim.save(filename+'.mp4', writer=writer1)

    # If it had to be saved as gif
    # anim.save('pplot1.gif', writer='imagemagick')






