import numpy as np
import random as rnd
import time
import math
"""
This code creates a 4x4 gameboard looking like this:

|-(0)-|-(1)-|-(2)-|-(3)|
|-(4)-|-(5)-|-(6)-|-(7)|
|-(8)-|-(9)-|-(10)-|-(11)|
|-(12)-|-(13)-|-(14)-|-(15)|

represented as arrays with indexes as seen above

there is a 2D-Array which stores the Q-values for each state(collumns) and possible action(rows)
there is a 1D-Array which stores the reward for each state
there is a replay_buffer storing all the action/state pairs leading up to a reset


the agent starts in state 12 which has a reward of 0

state 3 has a reward of 10, states 0 and 15 have a reward of -10
all the other states have a reward of 0
states 0, 3 and 15 are terminal states (agent will be reset to state 12)

the agent uses standard Q-learning and an n-step replaybuffer to determine the best action for each state

Q_new = Q_old + learning_rate*(R(s)+ 1/2V(s')+V(s))
with a learning_rate of 0.2, and V(s) being max_a(Q(s,a))

there is an epsylon-greedy strategie in place starting at epsylon = 1 (100% exploration)
after every reset the epsylon is decreased by 0.05 down to a minimum of 0.05 (5% exploration) 

After some time the agent will be able to move from state 12 to state 3
"""
Q_value_table = np.zeros((4, 16)) #Q-values (4 possible actions, 16 states)
Reward_table = [-10, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10] #Rewards
replay_buffer = [] #replaybuffer
current_field = 12 #startingposition
learning_rate = 0.2  # learning-rate
total_reward = 0
epsilon = 1 # epsylon-greedy strategy 

# edit UCB_value-table --> mark non existing actions --> those actions will never be chosen

Q_value_table[0,0] = -100
Q_value_table[3,0] = -100
Q_value_table[0,1] = -100
Q_value_table[0,2] = -100
Q_value_table[0,3]= -100
Q_value_table[1,3]= -100
Q_value_table[3,4]= -100
Q_value_table[1,7]= -100
Q_value_table[3,8]= -100
Q_value_table[1,11]= -100
Q_value_table[2,12]= -100
Q_value_table[3,12]= -100
Q_value_table[2,13]= -100
Q_value_table[2,14]= -100
Q_value_table[1,15]= -100
Q_value_table[2,15]= -100

while True:
    reset = False  # to check whether a reset has happened
    possible_actions = [] # up(0), right(1), down(2), left
    
    #check possible actions
    if current_field not in [0, 1, 2, 3]:  # no up
        possible_actions.append(0)
    if current_field not in [3, 7, 11, 15]:  # no right
        possible_actions.append(1)
    if current_field not in [12, 13, 14, 15]:  # no down
        possible_actions.append(2)
    if current_field not in [0, 4, 8, 12] :  # no left
        possible_actions.append(3)

    previous_field = current_field

    if Reward_table[current_field] in [10, -10]:  # reset
        current_field = 12
        reset = True
        print("RESET")
        if epsilon > 0.05:
            epsilon -= 0.05

    else:
        # exploration
        if rnd.random()<epsilon:
            action = rnd.choice(possible_actions)

        # next action based on Q-value
        else:
            action = np.argmax(Q_value_table[:, current_field])
        
        #calculate new field
        if action == 0:
            current_field-=4
        if action == 1:
            current_field+= 1
        if action == 2:
            current_field+= 4
        if action == 3:
            current_field-= 1

    total_reward += Reward_table[previous_field]

    
    if not reset:
        replay_buffer.insert(0, [action, previous_field]) # add action/state to replaybuffer
        print(replay_buffer)
    else:
        replay_buffer.insert(0, [0, previous_field])
        Q_value_table[:, previous_field] = Reward_table[previous_field]
        
        # update Q-value based on replaybuffer
        for i in range(len(replay_buffer)):
            if i == 1:
                Q_value_table[replay_buffer[i][0], replay_buffer[i][1]] = Q_value_table[replay_buffer[i][0], replay_buffer[i][1]] + learning_rate*(Reward_table[replay_buffer[i][1]] + 0.5*Q_value_table[0, replay_buffer[i-1][1]]-np.max(Q_value_table[replay_buffer[i][0], replay_buffer[i][1]]))
            elif i == 0:
                Q_value_table[:, replay_buffer[i][1]] = Reward_table[replay_buffer[i][1]]
            else:
                Q_value_table[replay_buffer[i][0], replay_buffer[i][1]] = Q_value_table[replay_buffer[i][0], replay_buffer[i][1]] + learning_rate*(Reward_table[replay_buffer[i][1]] + 0.5*np.max(Q_value_table[replay_buffer[i-1][0], replay_buffer[i-1][1]])-np.max(Q_value_table[replay_buffer[i][0], replay_buffer[i][1]]))

        replay_buffer = [] # empty replaybuffer
        print("TOTAL REWARD:" + str(total_reward))

    print(Q_value_table)
    print("CURRENT FIELD:"+str(current_field))
    time.sleep(0.4)
