import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from tqdm import tqdm
from utils import pairwise


def plot_main(data_x, data_y, filename=None, show=False):
    print("Producing information plane image")
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    for ix in tqdm(range(len(data_x))):
        for e in pairwise(zip(data_x[ix], data_y[ix])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)

    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()
    plt.cla()


def plot_one(data_x, data_y, filename=None, show=False):
    print("Producing information plane image")
    cmap = plt.get_cmap('gnuplot')
    for id, e in enumerate(pairwise(zip(data_x, data_y))):
        (x1, y1), (x2, y2) = e
        plt.plot((x1, x2), (y1, y2), linewidth=0.2, alpha=0.9, color=cmap(min(.9, 0.15 + id*0.1)))
        point_size = 300
        plt.scatter(x1, y1, s=point_size, alpha=0.9, color=cmap(min(.9, 0.15 + id*0.1)))
        plt.scatter(x2, y2, s=point_size, alpha=0.9, color=cmap(min(.9, 0.15 + id*0.1)))

    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()
    plt.cla()


def plot_bilayer(series, filename=None, show=False):
    print("Producing bilayer information image")
    series = np.array(series).swapaxes(0, 1)
    for ix, layer in enumerate(series):
        plt.plot(layer, label="{} -> {}".format(ix, ix+1))

    plt.xlabel("Time")
    plt.ylabel("Mutual Information")
    plt.title("Mutual Information between layer i and i+1")
    plt.legend(loc="upper left")
    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()
    plt.cla()


def plot_movie(data, movie_length, filename=None):
    print("Producing information plane movie")
    cmap = plt.get_cmap('gnuplot')
    figure, ax = plt.subplots()
    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')
    title = plt.title("epoch : 0", color='darkslategrey')
    frames = sorted(data.keys())

    def single_epoch(ix):
        for e in pairwise(zip(data[frames[ix]][0], data[frames[ix]][1])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=cmap(ix / len(data)), alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 50
            plt.scatter(x1, y1, s=point_size, color=cmap(ix / len(data)), zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=cmap(ix / len(data)), zorder=ix)
            title.set_text("  epoch: {}\n".format(frames[ix]))
        return figure

    movie = anim.FuncAnimation(figure, single_epoch, frames=len(data))

    if filename is not None:
        filename = filename + ".mp4"
        print("Saving movie to a file : ", filename)
        start = time.time()
        fps = max(3, int(len(data) / movie_length))
        print("fps {}".format(fps))
        writer = anim.writers['ffmpeg'](fps=fps)
        movie.save(filename, writer=writer, dpi=250)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    plt.cla()


def __select_frames(data_x, data_y, delta=0.6):
    """
    :param data_x: array of point`s x
    :param data_y: array of points`s y
    :param delta: distance
    :return: returns list of consecutive points with distance > delta.
    """
    delta = delta * delta
    assert(len(data_y) == len(data_x))
    frames = list(range(10))
    prev = frames[-1]

    def to_add(ia, ib):
        dist = 0

        def d2(x_a, y_a, x_b, y_b):
            return (x_a-x_b)*(x_a-x_b) + (y_a-y_b)*(y_a-y_b)
    
        for (xa, ya), (xb, yb) in zip(zip(data_x[ia], data_y[ia]), zip(data_x[ib], data_y[ib])):
            dist = max(dist, d2(xa, ya, xb, yb))
    
        return dist > delta

    for i in range(frames[-1], len(data_x) - 10):
        if to_add(prev, i):
            frames.append(i)
            prev = i

    for i in range(len(data_x) - 10, len(data_x)):
        frames.append(i)
    return frames






