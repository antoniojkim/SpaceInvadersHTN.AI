"""Training the model"""
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
    with tf.Session() as session:
        model = Model()
        done = False
        alpha = 1e-3
        gamma = 0.99
        target = None
        episode = 0
        curr_obs = env.reset()
        prev_obs = None
        net_reward = 0
        curr_reward = 0
        memory = []
        death_count = 0
        action_to_prob = [3, 2, 1, 1, 0]

        num_actions = 4
        max_episodes = 30000
        threshhold = 100
        Q = None

        for eps in range(max_episodes):

            if Q is None:

                epsilon_prob = [random.uniform(0, 1) for _ in range(num_actions)]

                action = choose_action(epsilon_prob)

            elif eps % 57 == 0:

                epsilon_prob = [random.uniform(0, 1) for _ in range(num_actions)]

                action = choose_action(epsilon_prob)

            else:
                prob = model.forward_pass(session, curr_obs.reshape([1, 185, 120, 1]))
                action = choose_action(prob)


            curr_obs, curr_reward, done, info = env.step(action)

            curr_obs = process_observations(curr_obs, prev_obs)

            memory.append((prev_obs, action, curr_reward, curr_obs, eps))

            while len(memory) > 100:

                rand_death_i = random.randint(0, len(memory) - 1)

                del memory[rand_death_i]

            if eps % threshhold == 0:

                print(eps, "EPISODE")

                rand_i = random.randint(0, len(memory) - 1)

                mem_sample = memory[rand_i]

                if Q is None:

                    target = mem_sample[2]
                    Q = np.random.random_sample((max_episodes, num_actions))

                else:

                    target = mem_sample[2] + gamma * Q[mem_sample[-1], max_Q(Q, mem_sample[-1])]
                    #print(target, "TARGET")
                model.train(session, training_data=mem_sample[-2], labels=Q[eps], target=target)

            Q[eps, action_to_prob[action]] += alpha * (target - Q[eps, action_to_prob[action]])

            env.render()

            prev_obs = curr_obs

            if info['ale.lives'] == 0:
                death_count = death_count + 1
                done = False
                env.reset()
                print(death_count, "DEATH COUNT")

        print(model.save_variables(session))



if __name__ == '__main__':

    main_loop()
