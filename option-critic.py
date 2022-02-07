
from fourrooms import FourRooms
from collections import defaultdict
from time import sleep

import numpy as np

# from IPython.display import clear_output
# import matplotlib.pyplot as plt

from utility import SoftmaxPolicy, SigmoidTermination, EpsGreedyPolicy, Critic, SigmoidTerminationBaseActions

env = FourRooms()
env.reset()

# clear_output(True)
# env.goal = (2,4)
# plt.imshow(env.render(show_goal=False), cmap='Blues')
# plt.axis('off')
# plt.show()

discount = 0.99

lr_term = 0.15
lr_intra = 0.15
lr_critic = 0.15

epsilon = 1e-1
eta = 0.020

temperature = 1e-2

# nruns = 10
nruns = 1

# nepisodes = 1000
nepisodes = 1000

nsteps = 200
# Number of options
noptions = 4



rng = np.random.RandomState(1234)

#adding doorways
possible_next_goals =[(6,2), (3,6), (7,9), (10,6)]

# History of steps and average durations
history = np.zeros((nruns, nepisodes, 2))

option_terminations_list = []

for run in range(nruns):

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.shape[0]

    # Following three belong to the Actor

    # 1. The intra-option policies - linear softmax functions
    option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]

    # 2. The termination function - linear sigmoid function
    option_terminations = [SigmoidTermination(rng, lr_term, nstates) for _ in range(noptions)]


    # 3. The epsilon-greedy policy over options
    policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)

    # Critic
    critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table, nstates, noptions, nactions)

    env.goal = (11,5)
    print('Goal: ', env.goal)

    c = 0
    for episode in range(nepisodes):

        # Change goal location after 1000 episodes
        # Comment it for not doing transfer experiments
        if episode%1000 == 0:
            env.goal = possible_next_goals[rng.choice(len(possible_next_goals))]
            print('New goal: ', env.goal)

        state = env.reset()

        option = policy_over_options.sample(state)
        # hardcoding actions according to options
        action = option_policies[option].sample(state)

        critic.cache(state, option, action)

        duration = 1
        option_switches = 0
        avg_duration = 0.0

        for step in range(nsteps):

            state, reward, done, _ = env.step(action)
            reward = reward + c

            # Termination might occur upon entering new state
            # if termination is true
            if option_terminations[option].sample(state):
                c = eta
                option = policy_over_options.sample(state)
                option_switches += 1
                avg_duration += (1.0 / option_switches) * (duration - avg_duration)
                duration = 1
            else:
                c = 0
            action = option_policies[option].sample(state)

            # Critic update
            critic.update_Qs(state, option, action, reward, done, option_terminations)

            # Intra-option policy update with baseline
            Q_U = critic.Q_U(state, option, action)
            Q_U = Q_U - critic.Q_Omega(state, option)
            option_policies[option].update(state, action, Q_U)

            # Termination condition update
            option_terminations[option].update(state, critic.A_Omega(state, option) + eta)

            duration += 1

            if done:
                break

        history[run, episode, 0] = step
        history[run, episode, 1] = avg_duration

    option_terminations_list.append(option_terminations)

    #Plot stuff
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    plt.title('run: %s' % run)
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.plot(np.mean(history[:run + 1, :, 0], axis=0))
    plt.grid(True)
    plt.subplot(122)
    plt.title('run: %s' % run)
    plt.xlabel('episodes')
    plt.ylabel('avg. option duration')
    plt.plot(np.mean(history[:run + 1, :, 1], axis=0))
    plt.grid(True)
    plt.show()

for run in range(nruns):

    termination_maps = [env.occupancy.astype('float64') for _ in range(noptions)]

    for option in range(noptions):
        for i in range(13):
            for j in range(13):
                state = (i,j)
                if termination_maps[option][i, j] == 0:
                    termination_maps[option][i, j] = option_terminations_list[run][option].pmf(state)

    print('Run: {}'.format(run))
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.title('option: 0', fontsize=20)
    plt.imshow(termination_maps[0], cmap='Blues')
    plt.axis('off')
    plt.subplot(142)
    plt.title('option: 1', fontsize=20)
    plt.imshow(termination_maps[1], cmap='Blues')
    plt.axis('off')
    plt.subplot(143)
    plt.title('option: 2', fontsize=20)
    plt.imshow(termination_maps[2], cmap='Blues')
    plt.axis('off')
    plt.subplot(144)
    plt.title('option: 3', fontsize=20)
    plt.imshow(termination_maps[3], cmap='Blues')
    plt.axis('off')
    plt.show()
    sleep(2)

# Turn off exploration while testing
policy_over_options.epsilon = 0
for option in range(noptions):
    option_policies[option].temperature = 1e-10

env = FourRooms()
track_actions = defaultdict()
nepisodes = 10

# test the agent
rng = np.random.RandomState(1234)
goal_reached_in_each_ep = []
for episode in range(nepisodes):
    track_actions[episode] = {0: np.full((13, 13), ""),
                              1: np.full((13, 13), ""),
                              2: np.full((13, 13), ""),
                              3: np.full((13, 13), ""),
                              4: np.full((13, 13), ""),
                              5: np.full((13, 13), ""),
                              6: np.full((13, 13), ""),
                              7: np.full((13, 13), "")}

    env.goal = possible_next_goals[rng.choice(len(possible_next_goals))]
    print('New goal: ', env.goal)

    state = env.reset()

    option = policy_over_options.sample(state)

    for step in range(nsteps):
        # fix state to only have first 2 not goal
        action = option_policies[option].sample(state)

        state, reward, done, _ = env.step(action)

        # Termination might occur upon entering new state
        if option_terminations[option].sample(state):
            option = policy_over_options.sample(state)

        track_actions[episode][option][state[0], state[1]]+=str(action)


        clear_output(True)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.title('episode: {}, step: {}'.format(episode, step), fontsize=20)
        plt.imshow(env.render(), cmap='Blues', )
        plt.axis('off')
        plt.subplot(122)
        plt.title('option: %s' % option, fontsize=20)
        plt.imshow(termination_maps[option], cmap='Blues')
        plt.axis('off')
        plt.show()

        if done:
            goal_reached_in_each_ep.append([episode, step])
            break

    print("Goal reached!")
    sleep(2)


