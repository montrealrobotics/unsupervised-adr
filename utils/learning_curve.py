from arguments import get_args
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.plot_utils import learning_curve

if __name__=='__main__':
    args = get_args()
    # load the model param
    selfplay_percent = [0.0, 1.0, 1.0]
    x_gen_labels = np.linspace(0, 199, 199)
    approach = ["udr", "adr", "unsupervised-default"]
    PLOTCOLORS = ['hotpink', 'red', 'darkolivegreen', 'hotpink', 'blue']
    save_plots = os.getcwd() + f'/plots/{args.env_name}/'
    if not os.path.isdir(save_plots):
        os.makedirs(save_plots)
    plt.rcParams["figure.figsize"] = (10, 6)
    SEED = [42, 41, 44, 43]

    for selfplay_index, a in enumerate(approach):
        mean, std = learning_curve(args, selfplay_percent, selfplay_index, a, SEED)
        mean = np.reshape(mean, (-1))
        std = np.reshape(std, (-1))
        plt.plot(x_gen_labels, mean, label=f'{a} - sp{selfplay_percent[selfplay_index]}',  alpha=0.7)
        plt.fill_between(x_gen_labels, mean + std/2, mean - std/2,  alpha=0.2)
    plt.title(f'Learning Curve for {args.env_name} | Mode : {args.mode}')
    plt.xlabel("Number of evaluations | 1 eval per 5000 timesteps")
    plt.ylabel("Average Distance")
    plt.ylim(0, 0.3)
    plt.xlim(0, 200)
    plt.legend()
    plt.savefig(f'{save_plots}/{args.mode}_generalization_sp{selfplay_percent[1]}.png', figsize=(10, 6))
    plt.show()