import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.getcwd() + '/saved_models/sp1.0polyak0.95-adr/'
epochs = 50
seeds = [123, 128, 256, 512, 156]
epoch = [0, 14, 24, 49]
for e in epoch:
    alice_gen = np.load(file_path + f'{128}/ResidualPushRandomizedEnv-v0/evals.npy', allow_pickle=True)
    print(alice_gen.shape)
    xlabels = np.logspace(np.log10(0.01), np.log10(0.5), num=10)
    plt.plot(xlabels, alice_gen[e, :], label='Epoch {}'.format(e + 1))

plt.plot(xlabels, np.ones(shape=xlabels.shape) * 0.9, linestyle=':', label='90% Success Rate')
plt.title('Generalization for different training frictions')
plt.xlabel('Friction')
plt.ylabel('Average Success Rate (n=20)')
plt.legend()
plt.show()