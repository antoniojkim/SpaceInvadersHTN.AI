import gym
import tensorflow as tf
import random
import numpy as np


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

        return 0

    elif max_index == 1:

        return 1

    elif max_index == 2:

        return 3

    elif max_index == 3:

        return 4


def decay_rewards(rewards, gamma):

    decayed_rewards = np.zeros_like(rewards)

    temp = 0

    for t in reversed(range(rewards.size(), 0, -1)):

        if rewards[t] != 0:
            temp = 0

        temp *= gamma

        temp += rewards[t]

        decayed_rewards[t] = temp

    return decayed_rewards


def max_Q(Q, state):

    max_action = 0

    max = Q[state, max_action]

    for action in range(Q.shape[1]):

        if max < Q[state, action]:

            max = Q[state, action]

            max_action = action

    return max_action


def main_loop():

    env = gym.make('SpaceInvaders-v0')

    done = False

    gamma = 0.99

    episode = 0
    curr_obs = env.reset()
    prev_obs = None
    net_reward = 0
    curr_reward = 0
    memory = []

    num_actions = 4
    max_episodes = 10000

    Q = None

    for eps in range(max_episodes):

        if Q is None:

            epsilon_prob = [random.uniform(0, 1) for _ in range(num_actions)]

            action = choose_action(epsilon_prob)

        else:

            action = max_Q(Q, eps)

        curr_obs, curr_reward, done, info = env.step(action)

        curr_obs = process_observations(curr_obs, prev_obs)
        memory.append((prev_obs, action, curr_reward, curr_obs, eps))

        if eps % 4 == 0:

            rand_i = random.randint(0, len(memory) - 1)

            mem_sample = memory[rand_i]

            if Q is None:

                target = mem_sample[2]

            else:

                target = mem_sample[2] + gamma * Q[mem_sample[-1], max_Q(Q, mem_sample[-1])]

        if Q is None:

            Q = np.zeros([max_episodes, num_actions])

        env.render()

        prev_obs = curr_obs

if __name__ == '__main__':

    main_loop()
