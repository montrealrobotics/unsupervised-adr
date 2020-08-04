import itertools
import numpy as np
import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from arguments import get_args

PLOTCOLORS = ['darkmagenta', 'orange', 'red', 'darkolivegreen', 'hotpink', 'blue']


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
        array = np.load(save_dir + f'{args.env_name}{args.mode}_evaluation_{args.sp_gamma}.npy', allow_pickle=True)
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
    values = []
    save_plots = os.getcwd() + f'/plots/{args.env_name}/'
    ticks = ["0-200K", "200K-400k", "400K-600k", "600K-800k", "800k-1000k"]
    steps = [(t + 1)* 199000 for t in range(5)]
    start = 0
    for st in steps:
        print(start, st)
        seed_ = []
        for s in seed:
            save_dir = osp.join(os.getcwd() + '/' + args.save_dir, "sp" + str(sp_index) + "polyak" +
                                str(args.polyak) + '-' + str(approach), str(s), args.env_name + '/')
            if not os.path.isfile(save_dir + f'alice_envs_{args.sp_gamma}.npy'):
                continue
            alice_envs = np.load(save_dir + f'alice_envs_{args.sp_gamma}.npy', allow_pickle=True)
            print(alice_envs.shape, s)
            seed_.extend(alice_envs[start:st, :])
        samplings = np.reshape(seed_, (-1, 1))
        print(samplings.shape)
        start = st
        values.append(samplings)
    
    x_ticks = np.linspace(0.1, 0.8, 20)
    x_ticks = np.around(x_ticks, 3)
    values = np.asarray(values)
    print(values.shape)
    values = values.squeeze(2)
#    poses = np.arange(len(x_ticks))

    bins = 20
#    h = plt.hist(values.T, bins=bins, stacked=True, label=[t for t in ticks], alpha=0.4, color=PLOTCOLORS[:5])
    h = plt.hist(np.reshape(values, (-1, 1)), bins=bins, label='samplings', alpha=0.4, color=PLOTCOLORS[:1])
    np.save('{}_sampling.npy'.format(args.env_name), values)
    plt.ylabel('Frequency')
    plt.xlabel('Randomization Coefficient')
#    x_tickpos = [0.65 * patch.get_width() + patch.getxy()[0] for patch in h]
#    plt.xticks(x_ticks)
    plt.legend(loc="upper right")
    plt.title(f'{args.env_name} | Range (0.05 - 2)')
    plt.savefig(f'{save_plots}/sampling_plot_{args.env_name}_{args.sp_gamma}_range(0.05)_2.png')
    return values


if __name__=='__main__':
    args = get_args()
    approach = ["unsupervised-adr"]
    sp_index = [1.0]
    PLOTCOLORS = ['hotpink', 'red', 'darkolivegreen', 'hotpink', 'blue']
    save_plots = os.getcwd() + f'/plots/{args.env_name}/'
    if not os.path.isdir(save_plots):
        os.makedirs(save_plots)
    plt.rcParams["figure.figsize"] = (10, 6)
    SEED = [i for i in range(85, 91)]
    sampling_plot(args, seed=SEED, sp_index=sp_index[0], approach=approach[0])

