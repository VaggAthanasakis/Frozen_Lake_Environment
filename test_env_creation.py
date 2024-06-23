import itertools
import random

action_keep = 0     # keep the same stock 
action_switch = 1   # switch to the other stock




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



def decimal_to_binary_array_without_Padding(decimal):
    binary_string = bin(decimal)[2:]
    binary_array = [int(bit) for bit in binary_string]
    return binary_array


def generate_environment(N):
    MAX_STATE_LENGTH = len(decimal_to_binary_array_without_Padding(N))
    states_for_each_stock = 2**N
    total_states = N * states_for_each_stock
    
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
        
        binary_string = bin(current_state)[2:]                 # Convert decimal to binary string        
        curr_state_array = [int(digit) for digit in binary_string] # Convert the binary string to a list of integers (0s and 1s)
        # We can now use the array to find if each stock is in high (1s) or low (0s) state
        # So We now know that we are at state {x,L,L,H....,H} with x the number of current stock
        
        curr_state_array = decimal_to_binary_array(current_state, MAX_STATE_LENGTH)
   
        #__Keep Stock _______________________
        for j in range (stock*2**N, ((stock+1)*2**N)): # for every possible transition when keeping the same stock
            state_to_trans = j - stock * states_for_each_stock          # value (H or L) of all of the stocks at the state we will transition to, in decimal form (0,1,2,3...)
            # binary_string = bin(state_to_trans)[2:]                     # convert to binary
            # trans_state_array = [int(digit) for digit in binary_string] # take each bit separately (0 for L and 1 for H)
            
            trans_state_array = decimal_to_binary_array(state_to_trans, MAX_STATE_LENGTH) # convert to binary and take each bit separately (0 for L and 1 for H)
            
            transitionProb = 1
            # stock_state_current= 1
            # stock_state_trans = 1
            print("Length: ",len(trans_state_array))
            print("trans_state_array: ",trans_state_array)
            print("curr_state_array: ",curr_state_array)
            
            for k in range(len(trans_state_array)):
                stock_state_trans = trans_state_array[k] # 0 or 1 // low or high
                # print("stock_state_trans: ",stock_state_trans)
                stock_state_current = curr_state_array[k] # 0 or 1 // low or high
                # print("stock_state_current: ",stock_state_trans)
                if(stock_state_current == 0 and stock_state_trans == 0):       # Pi_LL
                    transitionProb = transitionProb * pi[stock][0]
                elif(stock_state_current == 0 and stock_state_trans == 1):  # pi_LH
                    transitionProb = transitionProb * pi[stock][1]
                elif(stock_state_current == 1 and stock_state_trans == 0):  # pi_HL
                    transitionProb = transitionProb * pi[stock][2]
                else:                                                          # pi_HH
                    transitionProb = transitionProb * pi[stock][3]
            
            nextState = j
            reward = random.uniform(-0.02, 0.1)
            action_Keep.append((transitionProb,nextState,reward))
            
        SubDictionary[action_keep] = action_Keep
        SubDictionary[action_switch] = action_Switch
        P[i]=SubDictionary
    
    
    
    return P

P = generate_environment(2)
#print(P)
for key, value in P.items():
    print(f"{key}: {value}")















# def generate_environment(N):
#     # Total number of states
#     total_states = N * (2 ** N)
    
#     # Helper function to generate the state combinations
#     def state_combinations(n):
#         return list(itertools.product([0, 1], repeat=n))
    
#     # Generate all possible states
#     states = []
#     for stock in range(N):
#         for combination in state_combinations(N):
#             states.append((stock + 1, *combination))
    
#     # Define the probabilities and rewards
#     prob_keep = [9/20, 1/20, 9/20, 1/20]
#     reward_keep = [-0.02, -0.02, 0.1, 0.1]
#     prob_switch = [9/20, 1/20, 9/20, 1/20]
#     reward_switch = [0.01, 0.05, 0.01, 0.05]
    
#     # Initialize the environment dictionary
#     environment = {}
    
#     # Populate the environment
#     for i, state in enumerate(states):
#         stock_invested, *stock_states = state
#         env_state = {}
        
#         # Define action_keep transitions
#         action_keep = []
#         for j in range(4):
#             next_state = (stock_invested, *stock_states)
#             #if():

#             action_keep.append((prob_keep[j], states.index(next_state), reward_keep[j]))
        
#         # Define action_switch transitions
#         action_switch = []
#         next_invested_stock = (stock_invested % N) + 1
#         for j in range(4):
#             next_state = (next_invested_stock, *stock_states)
#             action_switch.append((prob_switch[j], states.index(next_state), reward_switch[j]))
        
#         env_state['action_keep'] = action_keep
#         env_state['action_switch'] = action_switch
        
#         environment[i] = env_state
    
#     return environment

# # # Example usage:
# # N = 2
# # environment = generate_environment(N)
# # print(environment)








# ######################

# def generate_environment_2(N, c):
#     # Define states
#     states = []
#     for stock in range(N):
#         for combination in itertools.product([0, 1], repeat=N):
#             states.append((stock, *combination))
    
#     # Define actions
#     actions = ["keep", "switch"]
    
#     # Generate random expected gains and transition probabilities
#     expected_gains = [random.uniform(-0.02, 0.1) for _ in range(N)]
#     transition_probs = [0.1 if i < N/2 else 0.5 for i in range(N)]
    
#     # Initialize environment dictionary
#     environment = {}
    
#     # Populate the environment
#     for i, state in enumerate(states):
#         current_stock, *stock_states = state
#         env_state = {}
        
#         action_keep = []
#         action_switch = []
        
#         for j in range(2 ** N):
#             new_states = list(itertools.product([0, 1], repeat=N))
            
#             for next_state in new_states:
#                 next_state_index = states.index((current_stock, *next_state))
                
#                 if next_state[current_stock] == 0:
#                     prob = 1 - transition_probs[current_stock]
#                 else:
#                     prob = transition_probs[current_stock]
                    
#                 if next_state == state:
#                     reward = expected_gains[current_stock]
#                 else:
#                     reward = 0
                    
#                 action_keep.append((prob, next_state_index, reward))
                
#                 next_stock = (current_stock + 1) % N
#                 next_state_index_switch = states.index((next_stock, *next_state))
#                 action_switch.append((prob, next_state_index_switch, reward - c))
                
#         env_state["keep"] = action_keep
#         env_state["switch"] = action_switch
        
#         environment[i] = env_state
    
#     return environment

# N = 2
# environment = generate_environment_2(N,0.8)
# print(environment)


