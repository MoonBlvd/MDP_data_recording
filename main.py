import numpy as np
import csv
import pdb

horizon = 3
_lambda = 0.9
s_trans = np.array([[0.997, 0.003],[0.204,0.796]]) # assume binary anomaly score
def init_state():
    R0 = 0
    mean_R = 0 # mean anomaly score
    length = 0 # buffer length
    rest_memo = 1000 # available memory
    #total_R = 0 # total anomaly score of this sequence
    #num_recorded_events = 0 # number of recorded anomalous events
    #num_anomaly_in_seq = 0 # number of anomalous frames in the current buffer
    utility = length + rest_memo
    return [R0,mean_R, length, rest_memo], utility

def state_update(s,a):

    length = s[2] + a[0] # update buffer length
    if length == 0:
        mean_R = s[1]
    else:
        mean_R = (s[1]*s[2] + s[0]*a[0])/length # update mean score of the buffer
    rest_memo = s[3] - a[0]#length
    # total_R = s[4] + s[0]*a[0]
    return [[0, mean_R, length, rest_memo], [1, mean_R, length, rest_memo]],s_trans[s[0]] # 0 is normal, 1 is anomalous

def compute_reward(s):
    # we want the sequence have higher mean anomaly score, but we also want a higher total anomaly score.
    # we want to record more anomalous events
    # we would like to record a long sequence for each event, but use less memory.

    reward = s[1] + s[2] + s[3] + s[1]*s[2]

    return reward

# def decision_tree(s,i):
#     inner_score_list = np.ones(len(action_list))
#     outer_score_list = []
#     for j,a in enumerate(action_list):
#         new_s = state_update(s,a) # should output list of possible new_s and their probs
#         new_r = compute_reward(new_s)
#         inner_score_list[j] = new_r
#         # pdb.set_trace()
#         if i < horizon:
#             outer_score_list = np.append(outer_score_list, new_r + _lambda* decision_tree(new_s,i+1))
#     if i < horizon:
#         return outer_score_list
#     return inner_score_list

def decision_tree(s,i):
    inner_score_list = np.zeros(len(action_list)) # there are two possible new states given state and action
    outer_score_list = []
    for j,a in enumerate(action_list):
        new_s, probs = state_update(s,a) # should output list of possible new_s and their probs
        tmp = np.zeros(len(action_list)**(horizon-i))
        for k, possible_s in enumerate(new_s):
            new_r = compute_reward(possible_s)
            inner_score_list[j] += probs[k]*new_r
            # pdb.set_trace()
            if i < horizon:
                # outer_score_list = np.append(outer_score_list, new_r + _lambda* decision_tree(possible_s,i+1))
                tmp += probs[k]*(new_r + _lambda* decision_tree(possible_s,i+1))
        outer_score_list = np.append(outer_score_list, tmp)
    if i < horizon:
        return outer_score_list
    return inner_score_list

if __name__ == "__main__":
    gamma = 0.9
    s, U = init_state()
    action_list = [[0,0], [1,0]]
    anomaly_score = [0,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0]

    for i,R in enumerate(anomaly_score):
        s[0] = R
        reward = compute_reward(s)
        # for a in action_list:
        print ("iter: ", i)
        print ("State: ", s)
        Q_value_list = reward + _lambda * decision_tree(s,1)
        print (Q_value_list)

        # find the optimal action
        max_Q = np.max(Q_value_list)
        max_idx = np.argmax(Q_value_list)
        optimal_action = action_list[int(max_idx/len(action_list)**(horizon-1))]
        print ("Optimal action: ", optimal_action)
        input("continue...")
        s,_ = state_update(s,optimal_action)
        s = s[0]

            # new_s = state_update(s, a)
            # r = compute_reward(new_s)
            # tmp_U = r +
