"""This is a trained model using saved weights"""
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

        rand = random.uniform(0, 1)

        if rand < 0.2222:

            return 1

        elif 0.2222 < rand < 0.6:

            return 3

        else:

            return 4



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

    main_loop("./fourset/Model1_Variables  2017-09-17  08.52.55.ckpt")

# ./Model1_Variables  2017-09-17  00.04.44.ckpt


# ./Oneset/Model1_Variables  2017-09-17  07.41.36.ckpt : oneset
# ./twoset/Model1_Variables  2017-09-17  07.49.46.ckpt : twoset just stays in one spot and shoots
# ./threeset/Model1_Variables  2017-09-17  07.57.12.ckpt : threeset fucking piece of trash
# ./fourset/Model1_Variables  2017-09-17  08.52.55.ckpt : fourset
