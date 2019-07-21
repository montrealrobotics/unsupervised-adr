import torch
from models import actor
from arguments import get_args
import gym
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

import rpl_environments

COLORS = ['red', 'blue', 'green', 'brown', 'teal']

if __name__ == '__main__':
    args = get_args()
    # load the model param
    args.save_dir = osp.join(args.save_dir, str(args.seed))
    xlabels = np.logspace(np.log10(0.01), np.log10(0.5), num=20)

    for i, friction in enumerate([0.05, 0.3, 0.5, 'default']):    
        a = np.load(osp.join(args.save_dir, 'friction{}'.format(friction), '{}'.format(args.env_name), 'friction_generalization.npy'), allow_pickle=True)
        evals = []

        for f, e in a:
            evals.append(e)

        m = np.mean(evals, axis=1)
        std = np.std(evals, axis=1) / 2

        if friction == 'default': friction = 'Default (0.18)'
        plt.plot(xlabels, m, color=COLORS[i], label='Friction {}'.format(friction))
        plt.fill_between(xlabels, m-std, m+std, facecolor=COLORS[i], alpha=0.05)

    plt.plot(xlabels, np.ones(shape=xlabels.shape) * 0.9, linestyle=':', label='90% Success Rate')

    plt.title('Generalization for different training frictions')
    plt.xlabel('Friction')
    plt.ylabel('Average Success Rate (n=20)')
    plt.legend()

    plt.show()
        





