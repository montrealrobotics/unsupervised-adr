import itertools
import numpy as np
import os
import os.path as osp


def learning_curve(args, selfplay_percent, selplay_index, approach, seed):
    '''
    args: arguments
    selfplay_percent: list of selfplay percentages (list)
    selplay_index: index of particular approach (integer)
    approach: list of different approaches ( Ex: ['udr', 'adr', 'unsupervised-default']
    seed: list of seed numbers
    return: returns mean and standard deviation across all seeds.
    '''
    default_dist = []
    for s in seed:
        a = []
        save_dir = osp.join(os.getcwd() + '/' + args.save_dir, "sp" + str(selfplay_percent[selplay_index]) + "polyak" +
                            str(args.polyak) + '-' + str(approach), str(s) + '/')
        array = np.load(save_dir + f'{args.env_name}{args.mode}_evaluation.npy', allow_pickle=True)
        for arr in array:
            a.append(arr)
        array = np.asarray(a)
        print(f'Array shape : {array.shape} | Seed : {s} | Approach : {approach} | selplay_index : {selplay_index}')
        default_dist.append([array[:199]])

    mean = np.mean(np.asarray(default_dist), axis=0)
    N = args.window_size
    mode = 'causal'
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


def sampling_plot(args, seed, sp_index, approach='adr'):

    for s in seed:
        save_dir = osp.join(args.save_dir, "sp" + str(sp_index) + "polyak" +
                            str(args.polyak) + '-' + str(approach), str(s), args.env_name + '/')
        print(save_dir)
        alice_envs = np.load(save_dir + f'alice_envs.npy', allow_pickle=True)
        envs = alice_envs
        envs = list(itertools.chain(*envs))
    list_ = np.reshape(envs, (-1, 1))
    return list_

