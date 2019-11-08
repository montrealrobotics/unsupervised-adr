import torch
from models import actor
from arguments import get_args
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import itertools


def learning_curve(selplay_index, approach, seed):
    default_dist = []

    for s in seed:
        a = []
        save_dir = osp.join(args.save_dir, "sp" + str(sp[selplay_index]) + "polyak" +
                            str(args.polyak) + '-' + str(approach), str(s), args.env_name + '/')
        array = np.load(save_dir + 'evaluations.npy', allow_pickle=True)
        for arr in array:
            a.append(arr)
        array = np.asarray(a[1:])
        print(f'Array shape : {array.shape} | Seed : {s} | Approach : {approach} | selplay_index : {selplay_index}')
        array = np.asarray(a[1:103])
        # print(array.shape)
        default_dist.append([array[:, 2]])
    # print(max(default_dist))
    mean = np.mean(np.asarray(default_dist), axis=0)
    N = 5
    mode = 'causal'
    # mean = np.convolve(np.reshape(mean, (mean.shape[1])), np.ones((N,)) / N, mode='valid')
    mean = smooth(np.reshape(mean, (mean.shape[1])), N, mode=mode)
    std = np.std(np.asarray(default_dist), axis=0)
    std = smooth(np.reshape(std, (std.shape[1])), N, mode=mode)
    return mean, std


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def sampling_plot(sp_index, approach='adr'):

    for s in SEED:
        save_dir = osp.join(args.save_dir, "sp" + str(sp_index) + "polyak" +
                            str(args.polyak) + '-' + str(approach), str(s), args.env_name + '/')
        print(save_dir)
        alice_envs = np.load(save_dir + f'alice_envs.npy', allow_pickle=True)
        envs = alice_envs
        envs = list(itertools.chain(*envs))

    list_ = np.reshape(envs, (-1, 1))
    return list_


if __name__=='__main__':
    args = get_args()
    # load the model param
    sp = [0.0, 1.0]
    # sp = [0.5, 0.5, 1.0]

    approach = ["udr", "adr"]
    PLOTCOLORS = ['hotpink', 'red', 'darkolivegreen', 'hotpink', 'blue']
    save_plots = os.getcwd() + f'/plots/{args.env_name}/'
    if not os.path.isdir(save_plots):
        os.makedirs(save_plots)
    plt.rcParams["figure.figsize"] = (10, 6)
    SEED = [31, 33, 35] # 31 - 0.15 | 32 : 0.5
    list_ = sampling_plot(sp_index='0.5')
    for i, a in enumerate(approach):
        print(i, a)
        mean, std = learning_curve(i, a, SEED)
        mean = np.reshape(mean, (-1))
        std = np.reshape(std, (-1))
        x_gen_labels = np.linspace(0, mean.shape[0], mean.shape[0])

        plt.plot(x_gen_labels, mean, label=f'{a} - sp{sp[i]}',  alpha=0.7)
        plt.fill_between(x_gen_labels, mean + std/2, mean - std/2,  alpha=0.2)
    plt.title(f'Learning Curve for {args.env_name} |Hard env')
    plt.xlabel("Number of evaluations | 1 eval per 5000 timesteps")
    plt.ylabel("Average Distance")
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{save_plots}/hard_generalization_sp{sp[1]}.png', figsize=(10, 10))

    plt.show()