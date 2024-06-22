# %% [markdown]
# # **Project 2: Stock Portfolio Optimization - Assignment 3**

# %% [markdown]
# ***Athanasakis - Fragkogiannis***

# %% [markdown]
# **Importing Libraries**

# %%
import numpy as np
import tkinter as tk #loads standard python GUI libraries
import numpy as np
import time
import random
from tkinter import *



# %% [markdown]
# **Environment for Question 1**

# %%
print("q1")

# %% [markdown]
# **Environment*for Question 2* 

# %%


# Create the environment
# We are modeling this environment using 8 states in the format: {stock_currently_holding,state_of_stock_1,state_of_stock_2}

action_keep = 0     # keep the same stock 
action_switch = 1   # switch to the other stock


P = {

    # State {1,L,L}
    0:{
        action_keep: [
             (9/20, 0, -0.02),    # probability: 9/20, next_State: {1,L,L}, Reward: -0.02
             (1/20, 1, -0.02),    # {1,L,H}
             (9/20, 2, +0.1),     # {1,H,L}
             (1/20, 3, +0.1)      # {1,H,H}
        ],

        action_switch:[
            (9/20, 4, +0.01),    # {2,L,L}
            (1/20, 5, +0.05),    # {2,L,H}
            (9/20, 6, +0.01),    # {2,H,L}
            (1/20, 7, +0.05)     # {2,H,H}
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
            (1/20, 4, +0.01),    # {2,L,L}
            (9/20, 5, +0.05),    # {2,L,H}
            (1/20, 6, +0.01),    # {2,H,L}
            (9/20, 7, +0.05)     # {2,H,H}
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
            (9/20, 4, +0.01),    # {2,L,L}
            (1/20, 5, +0.05),    # {2,L,H}
            (9/20, 6, +0.01),    # {2,H,L}
            (1/20, 7, +0.05)     # {2,H,H}
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
            (1/20, 4, +0.01),    # {2,L,L}
            (9/20, 5, +0.05),    # {2,L,H}
            (1/20, 6, +0.01),    # {2,H,L}
            (9/20, 7, +0.05)     # {2,H,H}
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
             (9/20, 0, -0.02),  # {1,L,L}
             (1/20, 1, -0.02),  # {1,L,H}
             (9/20, 2, +0.1 ),  # {1,H,L}
             (1/20, 3, +0.1 )   # {1,H,H}
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
            (1/20, 0, -0.02),  # {1,L,L}
            (9/20, 1, -0.02),  # {1,L,H}
            (1/20, 2, +0.1 ),  # {1,H,L}
            (9/20, 3, +0.1 )   # {1,H,H}
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
             (9/20, 0, -0.02),  # {1,L,L}
             (1/20, 1, -0.02),  # {1,L,H}
             (9/20, 2, +0.1 ),  # {1,H,L}
             (1/20, 3, +0.1 )   # {1,H,H}
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
             (1/20, 0, -0.02),  # {1,L,L}
             (9/20, 1, -0.02),  # {1,L,H}
             (1/20, 2, +0.1 ),  # {1,H,L}
             (9/20, 3, +0.1 )   # {1,H,H}
        ]
    }

}

reward_values = [-0.02, -0.02, 0.1, 0.1, 0.01, 0.05, 0.01, 0.05]
reward = np.array(reward_values)
holes = []



# %% [markdown]
# **Creating the environment for Question 3**

# %%
print("HI")

# %% [markdown]
# **Implementing Policy Iteration Algorithm**

# %%
# The next few lines are mostly for accounting
Tmax = 100000
size = len(P)
n = m = np.sqrt(size)
print(size)
Vplot = np.zeros((size,Tmax)) #these keep track how the Value function evolves, to be used in the GUI
Pplot = np.zeros((size,Tmax)) #these keep track how the Policy evolves, to be used in the GUI
t = 0


#this one is generic to be applied in many AI gym compliant environments

def policy_evaluation(pi, P, gamma = 1.0, epsilon = 1e-10):  #inputs: (1) policy to be evaluated, (2) model of the environment (transition probabilities, etc., see previous cell), (3) discount factor (with default = 1), (4) convergence error (default = 10^{-10})
    t = 0   #there's more elegant ways to do this
    prev_V = np.zeros(len(P)) # use as "cost-to-go", i.e. for V(s')
    while True:
        V = np.zeros(len(P)) # current value function to be learnerd
        for s in range(len(P)):  # do for every state
            for prob, next_state, reward in P[s][pi(s)]:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                V[s] += prob * (reward + gamma * prev_V[next_state])
        if np.max(np.abs(prev_V - V)) < epsilon: #check if the new V estimate is close enough to the previous one; 
            break # if yes, finish loop
        prev_V = V.copy() #freeze the new values (to be used as the next V(s'))
        t += 1
        Vplot[:,t] = prev_V  # accounting for GUI  
    return V


def policy_improvement(V, P, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
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
    
    while True:
        old_pi = {s: pi(s) for s in range(len(P))}  #keep the old policy to compare with new
        V = policy_evaluation(pi,P,gamma,epsilon)   #evaluate latest policy --> you receive its converged value function
        pi = policy_improvement(V,P,gamma)          #get a better policy using the value function of the previous one just calculated 
        
        t += 1
        Pplot[:,t]= [pi(s) for s in range(len(P))]  #keep track of the policy evolution
        Vplot[:,t] = V                              #and the value function evolution (for the GUI)
    
        if old_pi == {s:pi(s) for s in range(len(P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break
    print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
    return V,pi

n = 4
m = 4


#############################################################
###################### Question 2 ###########################

V_opt,P_opt = policy_iteration(P,gamma = 1)



# The following implements the GUI if you are going to use it on your own PC. It will not run as is on Colab. 
# I would recommend to maintain some sort of simple GUI like this for your own implementations as well...it might help you. But it is not mandatory
# If you want to use in Colab, comment out the code from this point on.


def P_to_text(a):
    if a == 0: return 'K' 
    if a == 1: return 'S'
    # if a == 2: return 'R'
    # if a == 3: return 'U'
    
    
for s in range(len(P)):
    print(P_opt(s))

frame_text_V = tk.Frame()
frame_V = tk.Frame(highlightbackground="blue", highlightthickness=2)
frame_text_P = tk.Frame()
frame_P = tk.Frame(highlightbackground="green", highlightthickness=2)
frame_text_R = tk.Frame()
frame_R = tk.Frame(highlightbackground="red", highlightthickness=2)



def submit():
    iter = int(e.get())
    rows = []

    for i in range(n):
        cols = []
        for j in range(m):
#            e = Entry(relief=GROOVE, master = frame_V)
            e2 = tk.Label(master = frame_V, text = ( '%f'  %(Vplot[i*m+j,iter])), font=("Arial", 14))
            e2.grid(row=i, column=j, sticky=N+S+E+W, padx=10, pady = 10)   
#            e.insert(END, '%f'  %(v[i,j]))
            cols.append(e2)    
        rows.append(cols)
    
    rows = []

    for i in range(n):
        cols = []
        for j in range(m):
#            e = Entry(relief=GROOVE, master = frame_V)
            if i*m+j in holes:
                e3 = tk.Label(master = frame_P, text = 'H', font=("Arial", 18))
            else:
                e3 = tk.Label(master = frame_P, text = P_to_text((Pplot[i*m+j,iter])), font=("Arial", 14))
            e3.grid(row=i, column=j, sticky=N+S+E+W, padx=10, pady = 10)            
#            e.insert(END, '%f'  %(v[i,j]))
            cols.append(e3)
        rows.append(cols)
         
rows = []
    
for i in range(n):
    cols = []
    for j in range(m):
        e2 = tk.Label(master = frame_R, text = ( '%f'  %(reward[i*m+j])), font=("Arial", 14))
        e2.grid(row=i, column=j, sticky=N+S+E+W, padx=10, pady = 10)
#        e2.insert(END, '%f'  %(r[i,j]))
        cols.append(e2)
    rows.append(cols)
    
    

label_V = tk.Label(master=frame_text_V, text="Value Function at Iteration:", font=("Arial", 18))
label_V.pack()
iter_btn=tk.Button(frame_text_V,text = 'Submit', command = submit, font=("Arial", 18))
iter_btn.pack()
e = Entry(relief=GROOVE, master = frame_text_V, font=("Arial", 18))
e.pack()



label_P = tk.Label(master=frame_text_P, text="Optimal Policy:", font=("Arial", 18))
label_P.pack()

label_R = tk.Label(master=frame_text_R, text="Reward Function:", font=("Arial", 18))
label_R.pack()

frame_text_V.pack()
frame_V.pack()
frame_text_P.pack()
frame_P.pack()
frame_text_R.pack()
frame_R.pack()


mainloop()







