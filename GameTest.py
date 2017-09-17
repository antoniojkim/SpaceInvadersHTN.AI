import random

import gym
import numpy as np
import tensorflow as tf

from Models import Model1 as Model

def process_observations(curr_observation, prev_observation=None):

    crop_horizontal = 20
    crop_top = 25
    curr_obs = curr_observation[crop_top:, crop_horizontal:curr_observation.shape[1] - crop_horizontal, 0]

    if prev_observation is None:

        return curr_obs

    else:

        new_obs = curr_obs - prev_observation

        return new_obs

def choose_action(probs):

    max_index = np.argmax(probs)

    if max_index == 0:

        return 4

    elif max_index == 1:

        return 3

    elif max_index == 2:

        return 1

    elif max_index == 3:

        return 0



def main_loop(path):

    env = gym.make('SpaceInvaders-v0')
    model = Model()

    model.load_variables(path)

    print(model.weights[0], "Loaded Weights")

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        max_episodes = 100000
        env.reset()
        Q = None
        prev_obs = None
        curr_obs, curr_reward, done, info = env.step(0)
        curr_obs = process_observations(curr_obs, prev_obs)

        for eps in range(max_episodes):

            prob = model.forward_pass(session, curr_obs.reshape([1, 185, 120, 1]))
            action = choose_action(prob)

            curr_obs, curr_reward, done, info = env.step(action)

            curr_obs = process_observations(curr_obs, prev_obs)

            prev_obs = curr_obs

            env.render()

            if done is True:

                done = False

                env.reset()

if __name__ == '__main__':

    main_loop("./Model1_Variables  2017-09-17  07.35.16.ckpt")

# ./Model1_Variables  2017-09-17  00.04.44.ckpt
