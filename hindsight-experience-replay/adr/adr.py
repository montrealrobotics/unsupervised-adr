import numpy as np

from common.discriminator_rewarder import DiscriminatorRewarder
from common.svpg.svpg import SVPG

class ADR:
    def __init__(
        self,
        nparticles,
        nparams,
        state_dim,
        action_dim,
        temperature,
        svpg_rollout_length,
        svpg_horizon,
        max_step_length,
        reward_scale,
        initial_svpg_steps,
        seed,
        discriminator_batchsz
    ):
        assert nparticles > 2

        self.nparticles = nparticles
        self.nparams = nparams

        self.svpg_rollout_length = svpg_rollout_length
        self.svpg_horizon = svpg_horizon
        self.initial_svpg_steps = initial_svpg_steps

        self.seed = seed
        self.svpg_timesteps = 0

        self.discriminator_rewarder = DiscriminatorRewarder(
            state_dim=state_dim,
            action_dim=action_dim,
            discriminator_batchsz=discriminator_batchsz,
            reward_scale=reward_scale,
        )

        self.svpg = SVPG(
            nparticles=nparticles,
            nparams=self.nparams,
            max_step_length=max_step_length,
            svpg_rollout_length=svpg_rollout_length,
            svpg_horizon=svpg_horizon,
            temperature=temperature,
        )

        self.parameter_settings = np.ones(
            (self.nparticles,
            self.svpg_horizon,
            self.svpg.svpg_rollout_length,
            self.svpg.nparams)
        ) * -1

    def score_trajectories(self, randomized_trajectories):
        rewards = np.zeros((self.nparticles, self.svpg.svpg_rollout_length))
 
        for i in range(self.nparticles):
            for t in range(self.svpg.svpg_rollout_length):
                # flatten and combine all randomized and reference trajectories for discriminator
                randomized_discrim_score_mean, _, _ = \
                    self.discriminator_rewarder.get_score(randomized_trajectories[i][t])

                rewards[i][t] = randomized_discrim_score_mean

        return rewards
            
    def step_particles(self):
        if self.svpg_timesteps >= self.initial_svpg_steps:
            # Get sim instances from SVPG policy
            simulation_instances = self.svpg.step()

            index = self.svpg_timesteps % self.svpg_horizon
            self.parameter_settings[:, index, :, :] = simulation_instances

        else:
            # Creates completely randomized environment
            simulation_instances = np.ones((self.nparticles,
                                            self.svpg.svpg_rollout_length,
                                            self.svpg.nparams)) * -1

        assert (self.nparticles, self.svpg.svpg_rollout_length, self.svpg.nparams) \
            == simulation_instances.shape

        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))

        self.svpg_timesteps += 1
        return simulation_instances

    def train(self, reference_trajectories, randomized_trajectories):
        rewards = self.score_trajectories(randomized_trajectories)
        self._train_particles(rewards)
        self._train_discriminator(reference_trajectories, randomized_trajectories)

    def _train_discriminator(self, reference_trajectories, randomized_trajectories):
        flattened_randomized = [randomized_trajectories[i][t] for i in range(self.nagents)]
        flattened_randomized = np.concatenate(flattened_randomized)

        flattened_reference = [reference_trajectories[i][t] for i in range(self.nagents)]
        flattened_reference = np.concatenate(flattened_reference)

        self.discriminator_rewarder.train_discriminator(
            flattened_reference, 
            flattened_randomized,
            iterations=len(flattened_randomized)
        )
    
    def _train_particles(self, rewards):
        self.svpg.train(rewards)

    