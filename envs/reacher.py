import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as et
import mujoco_py


class ReacherRandomizedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self.reference_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           'assets/reacher.xml')
        mujoco_env.MujocoEnv.__init__(self, self.reference_path, frame_skip=2)

        # randomization
        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []
        self._locate_randomize_parameters()

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        self.arm = self.root.find(".//geom[@name='link0']")
        self.body = self.root.find(".//body[@name='body1']")

    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)


    def _create_xml(self):
        # TODO: I might speed this up, but I think is insignificant w.r.t to the model/sim creation...
        self._randomize_armlength()
        # self._randomize_size()

        return et.tostring(self.root, encoding='unicode', method='xml')

    # TODO: I'm making an assumption here that 3 places after the comma are good enough, are they?
    def _randomize_armlength(self):
        length = self.dimensions[0].current_value
        self.arm.set('fromto', '0 0 0 {:3f} 0 0'.format(length))
        self.body.set('pos', '{:3f} 0 0'.format(length))