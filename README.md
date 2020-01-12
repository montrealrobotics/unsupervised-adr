# Unsupervised Active Domain Randomization 
This repository provides official code base for the paper "Unsupervised Active Domain Randomization for Goal Specified Environments".

## Experiments
We perform our experiments on ErgoReacher, a 4 DoF arm and ErgoPusher, a 3-DoF arm from both in simulation and real world.
### Important Flags
There are few important flags which differentiate various experiments. `--approach` specifies the approach ['udr' | 'unsupervised-default' | 'unsupervised-adr'], `--sp-percent` flag specifies the self-play percentage. For all out experiments we use either `--sp-percent=1.0` or `--sp-percent=0.0` depending on the `--approach`.  `--only-sp` specifies where the bob operates. It is `True` for `--approach=unsupervised-default` and `False` for `--approach=unsupervised-adr`. For all the reacher experiments we used `--n-params=8` and for pusher experiments we used `--n-params=1`.   

### Uniform Domain Randomization 
For `ErgoReacher` baseline experiments:

`python  ddpg_train.py  --sp-percent 0.0 --approach 'udr' --env_name='ErgoReacherRandomizedEnv-Headless-v0' --n-params=8 `

For `ErgoReacher` baseline experiments:

`python  ddpg_train.py  --sp-percent 0.0 --approach 'udr' --env_name='ErgoPusherRandomizedEnv-Headless-v0' --n-params=1`

### Unsupervised Default 
For `ErgoReacher` experiments:

`python  ddpg_train.py  --sp-percent 1.0 --approach 'unsupervised-default' --only-sp=True  --env_name='ErgoReacherRandomizedEnv-Headless-v0' --n-params=8 `

For `ErgoReacher` experiments:

`python  ddpg_train.py  --sp-percent 1.0 --approach 'unsupervised-default' --only-sp=True  --env_name='ErgoPusherRandomizedEnv-Headless-v0' --n-params=1`

### Unsupervised Active Domain Randomization
For `ErgoReacher` experiments:

`python  ddpg_train.py  --sp-percent 1.0 --approach 'unsupervised-adr' --only-sp=True  --env_name='ErgoReacherRandomizedEnv-Headless-v0' --n-params=8`

For `ErgoReacher` experiments:

`python  ddpg_train.py  --sp-percent 1.0 --approach 'unsupervised-adr' --only-sp=True  --env_name='ErgoPusherRandomizedEnv-Headless-v0' --n-params=1`

## Reference
coming soon.

Built by [@Sharath](https://sharathraparthy.github.io/) and [@Bhairav Mehta](https://bhairavmehta95.github.io/)