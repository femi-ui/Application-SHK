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
there is a 2D-Array which stores the number of times an action has been chosen 
there is a 2D-Array which stores the UCB1-value for each action
there is a 1D-Array which stores the reward for each state

the agent starts in state 12 which has a reward of 0

state 3 has a reward of 10, state 0 and 15 have a reward of -10
all the other states have a reward of 0
states 0,3 and 15 are terminal states (agent will be reset to state 12)

the agent uses standard Q-learning and UCB1 to determine the best action for each state

Q_new = Q_old + learning_rate*(R(s)+ 1/2V(s')+V(s))
with a learning_rate of 0.2, and V(s) being max_a(Q(s,a))

UCB_value = Q+1*sqrt(2*ln(t)/n)
with t being the total number of actions taken and n being the number of times the specific action has been taken

After some time the agent will be able to move from state 12 to state 3
"""

Q_value_table = np.zeros((4, 16)) #Q-values (4 possible actions, 16 states)
Reward_table = [-10, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10] #Rewards
action_counter = np.zeros((4, 16)) #action_counter
UCB_values = np.zeros((4,16))
current_field = 12 #startingposition
learning_rate = 0.2  # learning-rate
total_actions = 0
total_reward = 0
exploration = 0 # to manage exploration before UCB1 takes over

# edit UCB_value-table --> mark non existing actions --> those actions will never be chosen

UCB_values[0,0] = -100
UCB_values[3,0] = -100
UCB_values[0,1] = -100
UCB_values[0,2] = -100
UCB_values[0,3]= -100
UCB_values[1,3]= -100
UCB_values[3,4]= -100
UCB_values[1,7]= -100
UCB_values[3,8]= -100
UCB_values[1,11]= -100
UCB_values[2,12]= -100
UCB_values[3,12]= -100
UCB_values[2,13]= -100
UCB_values[2,14]= -100
UCB_values[1,15]= -100
UCB_values[2,15]= -100

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
    if current_field not in [0, 4, 8, 12]:  # no left
        possible_actions.append(3)

    previous_field = current_field

    if Reward_table[current_field] in [10, -10]:  # reset
        current_field = 12
        reset = True
        print("RESET")
        exploration += 1

    else:
        if exploration<10: # 5 resets before UCB1 takes over
            action = rnd.choice(possible_actions)
            
        # next action based on UCB-value
        else:
            action = np.argmax(UCB_values[:, current_field])
        
        if action == 0:
            current_field-=4
        if action == 1:
            current_field+= 1
        if action == 2:
            current_field+= 4
        if action == 3:
            current_field-= 1
        total_actions += 1
        action_counter[action, previous_field] += 1

    total_reward += Reward_table[previous_field]

    # update Q-value and
    if not reset:
        Q_value_table[action, previous_field] = Q_value_table[action, previous_field] + learning_rate*(Reward_table[previous_field] + 0.5*np.max(Q_value_table[:, current_field])-np.max(Q_value_table[action, previous_field]))
        # calculate UCB
        UCB_values[action, previous_field] = Q_value_table[action, previous_field] + np.sqrt(
            np.log(total_actions) / action_counter[action, previous_field])

    else:
        Q_value_table[:, previous_field] = Reward_table[previous_field]
        print("TOTAL REWARD:" + str(total_reward))

    print(UCB_values)
    print("CURRENT FIELD:"+str(current_field))
    time.sleep(0.2)
