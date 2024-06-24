import numpy as np
import tkinter as tk #loads standard python GUI libraries
import numpy as np
import time
import random
from tkinter import *



# Create the three different environments
# We are modeling this environment using 8 states in the format: {stock_currently_holding,state_of_stock_1,state_of_stock_2}

action_keep = 0     # keep the same stock 
action_switch = 1   # switch to the other stock

# This environment is used for the question 1 where we need to demonstrate that the optimal 
# policy is always to stay with the stock we already have invested
fee = -0.9
# r1H = 2*r2L
# in this case r1.h=0.1 // r2.H= 0.05 // r1.L = -0.02 // r2.L = 0.01
# we have used a large transaction fee so that the best policy will always be to keep using the same stock
P1 = {

    # State {1,L,L}
    0:{
        action_keep: [
             (9/20, 0, -0.02),    # probability: 9/20, next_State: {1,L,L}, Reward: -0.02
             (1/20, 1, -0.02),    # {1,L,H}
             (9/20, 2, +0.1),     # {1,H,L}
             (1/20, 3, +0.1)      # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,L,H}
    1:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05 + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,L}
    2:{
        action_keep: [
             (9/20, 0, -0.02),  # {1,L,L}
             (1/20, 1, -0.02),  # {1,L,H}
             (9/20, 2, +0.1 ),  # {1,H,L}
             (1/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,H}
    3:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch: [
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05 + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {2,L,L}
    4:{
        action_keep: [
             (9/20, 4,  +0.01),    # {2,L,L}
             (1/20, 5,  +0.05),    # {2,L,H}
             (9/20, 6,  +0.01),    # {2,H,L}
             (1/20, 7,  +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1  + fee),  # {1,H,L}
             (1/20, 3, +0.1  + fee)   # {1,H,H}
        ]
    },

    # State {2,L,H}
    5:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
            (1/20, 0, -0.02 + fee),  # {1,L,L}
            (9/20, 1, -0.02 + fee),  # {1,L,H}
            (1/20, 2, +0.1  + fee),  # {1,H,L}
            (9/20, 3, +0.1  + fee)   # {1,H,H}
        ]
    },

    # State {2,H,L}
    6:{
        action_keep: [
             (9/20, 4, +0.01),    # {2,L,L}
             (1/20, 5, +0.05),    # {2,L,H}
             (9/20, 6, +0.01),    # {2,H,L}
             (1/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1  + fee),  # {1,H,L}
             (1/20, 3, +0.1  + fee)   # {1,H,H}
        ]
    },

    # State {2,H,H}
    7:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (1/20, 0, -0.02 + fee),  # {1,L,L}
             (9/20, 1, -0.02 + fee),  # {1,L,H}
             (1/20, 2, +0.1  + fee),  # {1,H,L}
             (9/20, 3, +0.1  + fee)   # {1,H,H}
        ]
    }

}


# This environment implements the stocks environment from the midterm
# It is used for the question 2 where we need to demonstrate that the optimal policy
# for some of the states is to switch and in some others to stay
fee = -0.01
P2 = {
    
    # State {1,L,L}
    0:{
        action_keep: [
             (9/20, 0, -0.02),    # probability: 9/20, next_State: {1,L,L}, Reward: -0.02
             (1/20, 1, -0.02),    # {1,L,H}
             (9/20, 2, +0.1),     # {1,H,L}
             (1/20, 3, +0.1)      # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,L,H}
    1:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05 + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,L}
    2:{
        action_keep: [
             (9/20, 0, -0.02),  # {1,L,L}
             (1/20, 1, -0.02),  # {1,L,H}
             (9/20, 2, +0.1 ),  # {1,H,L}
             (1/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01 + fee),    # {2,L,L}
            (1/20, 5, +0.05 + fee),    # {2,L,H}
            (9/20, 6, +0.01 + fee),    # {2,H,L}
            (1/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {1,H,H}
    3:{
        action_keep: [
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ],

        action_switch: [
            (1/20, 4, +0.01 + fee),    # {2,L,L}
            (9/20, 5, +0.05  + fee),    # {2,L,H}
            (1/20, 6, +0.01 + fee),    # {2,H,L}
            (9/20, 7, +0.05 + fee)     # {2,H,H}
        ]
    },

    # State {2,L,L}
    4:{
        action_keep: [
             (9/20, 4,  +0.01),    # {2,L,L}
             (1/20, 5,  +0.05),    # {2,L,H}
             (9/20, 6,  +0.01),    # {2,H,L}
             (1/20, 7,  +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1 + fee),  # {1,H,L}
             (1/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,L,H}
    5:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
            (1/20, 0, -0.02 + fee),  # {1,L,L}
            (9/20, 1, -0.02 + fee),  # {1,L,H}
            (1/20, 2, +0.1 + fee),  # {1,H,L}
            (9/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,H,L}
    6:{
        action_keep: [
             (9/20, 4, +0.01),    # {2,L,L}
             (1/20, 5, +0.05),    # {2,L,H}
             (9/20, 6, +0.01),    # {2,H,L}
             (1/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (9/20, 0, -0.02 + fee),  # {1,L,L}
             (1/20, 1, -0.02 + fee),  # {1,L,H}
             (9/20, 2, +0.1 + fee),  # {1,H,L}
             (1/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    },

    # State {2,H,H}
    7:{
        action_keep: [
             (1/20, 4, +0.01),    # {2,L,L}
             (9/20, 5, +0.05),    # {2,L,H}
             (1/20, 6, +0.01),    # {2,H,L}
             (9/20, 7, +0.05)     # {2,H,H}
        ],

        action_switch:[
             (1/20, 0, -0.02 + fee),  # {1,L,L}
             (9/20, 1, -0.02 + fee),  # {1,L,H}
             (1/20, 2, +0.1 + fee),  # {1,H,L}
             (9/20, 3, +0.1 + fee)   # {1,H,H}
        ]
    }

}


# This environment implements the generic scenario of question 3 where for every stock
# ri_H,ri_L are chosen uniformly in [-0.02, 0.1] and transition probabilities pi_HL, pi_LH 
# are equal to 0.1 for half the stocks and 0.5 for the other half.

# Since every stock can have two price states, the number of total states in the MDP 
# we are creating will be = NumOfStoscks*2^numOfStocks


def decimal_to_binary_array(decimal, length):
    
    # Convert decimal to binary string (strip '0b' prefix)
    binary_string = bin(decimal)[2:]
    
    # Determine padding length
    padding_length = max(0, length - len(binary_string))
    
    # Pad binary string with leading zeros if needed
    padded_binary_string = '0' * padding_length + binary_string
    
    # Convert padded binary string to list of binary digits
    binary_array = [int(bit) for bit in padded_binary_string]
    
    return binary_array


# Function that generates the environment of N stocks dynamically, with a transaction fee  
def generate_environment(N,fee):
    
    states_for_each_stock = 2**N
    total_states = N * states_for_each_stock
    max_state_length = N
    
    P = {}
    pi = []
    #Creating transition probabilities for the keep action
    #of EACH stock
    for i in range(0,N):
        if(i < N/2):
            # pi_HL = pi_LH = 0.1 | # pi_HH = pi_LL = 0.9
            row = [0.9,0.1,0.1,0.9] #[LL,LH,HL,HH]            
        else:
            # pi_HL = pi_LH = 0.5 | # pi_HH = pi_LL = 0.5
            row = [0.5,0.5,0.5,0.5] #[LL,LH,HL,HH]       
        pi.append(row)    

    
    for i in range(0, total_states):
        SubDictionary={}
        action_Keep = [] 
        action_Switch = []

        # find what stock we are reffering to 
        # Stock ids start from 0
        stock = i // states_for_each_stock
        
        ##########################
        # We define states of L and H with binary ids
        # For example for 2 stocks this stranslation occurs:
        # LL -> 0,0 -> 0
        # LH -> 0,1 -> 1
        # HL -> 1,0 -> 2
        # HH -> 1,1 -> 3
        # The binary ids are then translated to decimals so that 
        # we can use them in code
        ##########################
        
        current_state = i - stock * states_for_each_stock # find where this specific stock starts at the total_states environment
                                                          # this is necessary to calculate the transition probabilities
        
        # Convert decimal to binary string        
        # Convert the binary string to a list of integers (0s and 1s)        
        curr_state_array = decimal_to_binary_array(current_state, max_state_length)
        # We can now use the array to find if each stock is in high (1s) or low (0s) state
        # So We now know that we are at state {x,L,L,H....,H} with x the number of current stock
   
        #__Keep Stock ________________________________________________________________________________________________________________
        for j in range (stock*2**N, ((stock+1)*2**N)): # for every possible transition when keeping the same stock
            state_to_trans = j - stock * states_for_each_stock          # value (H or L) of all of the stocks at the state we will transition to, in decimal form (0,1,2,3...)
            trans_state_array = decimal_to_binary_array(state_to_trans, max_state_length) # convert to binary and take each bit separately (0 for L and 1 for H)
            
            transitionProb = 1
            
            for k in range(len(trans_state_array)):
                stock_state_trans = trans_state_array[k] # 0 or 1 // low or high                
                stock_state_current = curr_state_array[k] # 0 or 1 // low or high
                
                if(stock_state_current == 0 and stock_state_trans == 0):       # Pi_LL
                    transitionProb = transitionProb * pi[stock][0]
                elif(stock_state_current == 0 and stock_state_trans == 1):     # pi_LH
                    transitionProb = transitionProb * pi[stock][1]
                elif(stock_state_current == 1 and stock_state_trans == 0):     # pi_HL
                    transitionProb = transitionProb * pi[stock][2]
                else:                                                          # pi_HH
                    transitionProb = transitionProb * pi[stock][3]
            
            nextState = j
            #reward = random.uniform(-0.02, 2)
            reward = random.uniform(-0.02, 0.1)
            action_Keep.append((transitionProb,nextState,reward))
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #fee = 0
        #__Switch Stock ________________________________________________________________________________________________________________
        for j in range (0, total_states): # for every possible transition when keeping the same stock
            trans_stock = j // states_for_each_stock
            
            if(trans_stock == stock):     # check if the transition stock is the same as the stock we start from
                continue                  # we have already handle this situation above so we move on
             
             
            trans_state = j - trans_stock * states_for_each_stock
            trans_state_array = decimal_to_binary_array(trans_state, max_state_length)
            transitionProb = 1
            
            for k in range(len(trans_state_array)):
                stock_state_trans = trans_state_array[k] # 0 or 1 // low or high                
                stock_state_current = curr_state_array[k] # 0 or 1 // low or high
                
                if(stock_state_current == 0 and stock_state_trans == 0):       # Pi_LL
                    transitionProb = transitionProb * pi[stock][0]
                elif(stock_state_current == 0 and stock_state_trans == 1):     # pi_LH
                    transitionProb = transitionProb * pi[stock][1]
                elif(stock_state_current == 1 and stock_state_trans == 0):     # pi_HL
                    transitionProb = transitionProb * pi[stock][2]
                else:                                                          # pi_HH
                    transitionProb = transitionProb * pi[stock][3] 
                    
            nextState = j
            #reward = random.uniform(-0.02, 2) - fee
            reward = random.uniform(-0.02, 0.1) - fee
            action_Switch.append((transitionProb,nextState,reward))   
        
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        SubDictionary[action_keep] = action_Keep
        SubDictionary[action_switch] = action_Switch
        P[i]=SubDictionary
    
    
    
    return P



Tmax = 100000000
t = 0


#this one is generic to be applied in many AI gym compliant environments

def policy_evaluation(pi, P, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
    #print("in policy EVALUATION")
    t = 0   #there's more elegant ways to do this
    prev_V = np.zeros(len(P)) # use as "cost-to-go", i.e. for V(s')
    while True:
        V = np.zeros(len(P)) # current value function to be learnerd
        for s in range(len(P)):  # do for every state
            for prob, next_state, reward in P[s][pi(s)]:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                V[s] = np.int64(V[s] + prob * (reward + gamma * prev_V[next_state]))
        if np.max(np.abs(prev_V - V)) < epsilon: #check if the new V estimate is close enough to the previous one;     
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
        t += 1
    return V


def policy_improvement(V, P, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
    #print("in policy IMPROVEMENT")
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64) #create a Q value array
    for s in range(len(P)):        # for every state in the environment/model
        for a in range(len(P[s])):  # and for every action in that state
            for prob, next_state, reward in P[s][a]:  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                Q[s][a] += prob * (reward + gamma * V[next_state])
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
    # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
    # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself
    
    return new_pi

# policy iteration is simple, it will call alternatively policy evaluation then policy improvement, till the policy converges.

def policy_iteration(P, gamma = 1.0, epsilon = 1e-10):
    t = 0
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))     # start with random actions for each state  
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
    print("Policy in first iteration:")
    print_policy(pi,len(P))
    print("\n")
    while True:
        old_pi = {s: pi(s) for s in range(len(P))}  #keep the old policy to compare with new
        V = policy_evaluation(pi,P,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
        pi = policy_improvement(V,P,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1    
        if old_pi == {s:pi(s) for s in range(len(P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('Converged after %d Policy Iterations' %t) #keep track of the number of (outer) iterations to converge
    return V,pi


# Function to print policy
def print_policy(policy, num_states=8):
    for s in range(num_states):
        print(f"State {s}: Action {policy(s)}")
        


#############################################################
###################### Question 1 ###########################
print("before policy_iteration ")
gamma = 0
V_opt1,P_opt1 = policy_iteration(P1,gamma)
print("\nPolicy after optimization:")
print_policy(P_opt1)




#############################################################
###################### Question 2 ###########################
print("Question 2")
print("before policy_iteration ")
gamma = 0.9
V_opt2,P_opt2 = policy_iteration(P2,gamma)
print("\nPolicy after optimization:")
print_policy(P_opt2)


#############################################################
###################### Question 3 ###########################
print("Question 3")
print("before policy_iteration ")
fee = 0.03
number_of_stocks = 5
P3 = generate_environment(number_of_stocks,fee)
gamma = 0.8
V_opt3,P_opt3 = policy_iteration(P3,gamma)
print("\nPolicy after optimization:")
print_policy(P_opt3,len(P3))






