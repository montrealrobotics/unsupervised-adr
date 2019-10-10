import torch
from models import actor
from arguments import get_args
import gym
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.interpolate import spline

args = get_args()


def plot_lc():
    xlabel_learning = np.linspace(1, 50, 2500)
    x_gen_labels = np.linspace(0, 50, 1)
    epoch = [0, 9, 19, 29, 39, 49]
    sp = [0.0, 0.5]
    approach = ["udr", "adr"]
    SEED = [36, 37, 38, 39]
    PLOTCOLORS = ['darkmagenta', 'orange', 'red', 'darkolivegreen', 'hotpink', 'blue']

    learning_params = ["Default Env Success", "Default Env Average Distance", "Hard Env Success", "Hard Env Average Distance"]
    save_dir = osp.join(args.save_dir, "sp" + str(sp[0]) + "polyak" +
                        str(args.polyak) + '-' + str(approach[0]), str(24), args.env_name + '/')
    learning = np.load(os.getcwd() + '/' + save_dir + 'evaluations.npz')
    for i, keys in enumerate(learning.keys()):
        print(keys)
        for idx, app in enumerate(approach):
            learnings = []
            seed = []
            for s in SEED:
                save_dir = osp.join(args.save_dir, "sp" + str(sp[idx]) + "polyak" +
                                    str(args.polyak) + '-' + str(approach[idx]), str(s), args.env_name + '/')
                learning = np.load(os.getcwd() + '/' + save_dir + 'evaluations.npz')
                key = learning[keys]
                print(len(key))
                learnings.append(key)
                seed.extend(learnings)
            print(np.reshape(np.asarray(seed), (-1, 1)).shape)
            learnings_mean = np.mean(learnings, axis=0)
            learnings_std = np.reshape(np.std(learnings, axis=0), (-1))
            y = np.convolve(np.reshape(learnings_mean, (-1)), np.ones(10) / 10)
            plt.plot(xlabel_learning, y[5:2505], label= f"{app} :{keys}", color=PLOTCOLORS[idx])
            plt.fill_between(xlabel_learning, np.reshape(learnings_mean, (-1))
                             - learnings_std / 2, np.reshape(learnings_mean, (-1)) + learnings_std / 2,
                             facecolor=PLOTCOLORS[idx], alpha=0.8)
            plt.xlim(0, 50)
            # plt.ylim(45, 55)
            plt.title(f"Learning curve (ErgoPush Environment) : {learning_params[i]}")
            plt.xlabel("Epoch")
            plt.ylabel(f'{learning_params[i]}')
            plt.legend()
        plt.savefig(f'{keys}.png')
        plt.show()
        # plt.clf()
    plt.close()

plot_lc()