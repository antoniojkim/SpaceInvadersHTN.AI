import gym
import matplotlib.pyplot as plt


def process_observations(curr_observation, prev_observation=None):
    crop_horizontal = 20
    crop_top = 25
    curr_obs = curr_observation[crop_top:, crop_horizontal:curr_observation.shape[1] - crop_horizontal, 0]

    if prev_observation is None:

        return curr_obs

    else:

        new_obs = curr_obs - prev_observation

        return new_obs

def show_change_obs(obs):

    plt.imshow(obs)
    plt.show()

def main_loop():

    env = gym.make('SpaceInvaders-v0')

    done = False

    episode = 0
    curr_obs = env.reset()
    prev_obs = None
    net_reward = 0
    curr_reward = 0

    while done is not True:

        curr_obs, curr_reward, done, info = env.step(1)

        print(curr_reward, done)
        curr_obs = process_observations(curr_obs, prev_obs)
        show_change_obs(curr_obs)
        env.render()

        prev_obs = curr_obs

if __name__ == '__main__':

    main_loop()
