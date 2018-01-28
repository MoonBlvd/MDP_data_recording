import numpy as np
import csv
import pdb
from process_warning_anomaly_data import *
from compressor import simpleCompress
import cv2

horizon = 3
_lambda = 0.9
# s_trans = np.array([[0.997, 0.003],[0.204,0.796]]) # assume binary anomaly score
num_warning = 3
num_anomaly = 4
num_WA = num_warning + num_anomaly
min_frames = 20
max_memo = 10000 # memory threshold in KBs
action_list = [0,1,2,3]

warning_anomaly_state_space = 2**num_warning * 3**num_anomaly # 3 binary warnings and 4 abstracted anomaly scores
WA_trans = np.eye(warning_anomaly_state_space) # identity matrix for transition

rec_trans_1 = np.array([[0,1],[0,1]]) # if action is record
rec_trans_2 = np.array([[1,0],[1,0]]) # if action is not record
# freq_trans =
# memo_trans_1 = np.array([])
# record_trans = np.kron()
# not_record_trans =

# summarize frequencies of each state.
freq_ctr = np.zeros(warning_anomaly_state_space) # a (648,) array to record frequency of each state
freq_threshold = 100 # threshold of frequency of warning and anomalies.
memory_cost = 0

ext = '.jpeg'
# weighting
w = [1,2,1,1]
'''
load pre-generated WA_state list
'''
state_list = []
with open('state_list.txt') as f:
    lines = f.readlines()
    for line in lines:
        state_list.append(line.strip())



def init_state():
    warning_anomaly_state = np.zeros(num_warning+num_anomaly).astype(int)
    ambient_state_discrete_i = compute_WA_state_index(warning_anomaly_state)
    ambient_state_discrete_u = compute_ambient_state_u(warning_anomaly_state)
    recording = 0 # not recoding at the beginning
    memo_full = 0 # memory is not full at the beginning
    frequent_anomaly = 0 # new anomaly is not frequent
    #total_R = 0 # total anomaly score of this sequence
    #num_recorded_events = 0 # number of recorded anomalous events
    #num_anomaly_in_seq = 0 # number of anomalous frames in the current buffer
    utility = 0
    s = {}
    s['i'] = int(ambient_state_discrete_i)
    s['u'] = ambient_state_discrete_u
    s['rec'] = int(recording)
    s['memo'] = memo_full
    s['freq'] = np.zeros(warning_anomaly_state_space).astype(int)
    return s, utility

def compute_WA_state_index(warning_anomaly_state):
    WA_state_in_str = ''
    for WA_state in warning_anomaly_state:
        WA_state_in_str += str(WA_state)
    WA_state_idx = state_list.index(WA_state_in_str)
    return WA_state_idx

def compute_WA_state_array(WA_state_idx):
    state_string = state_list[WA_state_idx]
    state_array = []
    for i in state_string:
        state_array.append(int(i))
    return np.array(state_array).astype(int)

def state_update(s,a,memo_cost,img):
    '''
    :param s: the whole state vector 
    :param a: the action, a = 0 (no record) or 1 (record)
    :return: all possible states and their probabilities
    '''
    '''Read WA_state and find possible next WA_state'''
    WA_state_idx = s['i']
    possible_state_idx = np.where(WA_trans[WA_state_idx, :] != 0)[0]
    probabilities = WA_trans[WA_state_idx, possible_state_idx]

    '''Find possible system state given state and action'''
    # rec_state = s[num_WA] ^ a # use XOR to find whether the recording state is flipped.

    s['rec'] = a # the recording state depends on the previous action

    # if a == 0 and prev_a == 1:
    #     rec_state = 1 # stop recording
    # elif a == 1 and prev_a == 0:
    #     rec_state = -1 # start recording
    # else:
    if a: # if record, check whether memory exceed the max.
        s['memo'] += compressor.run_opencv_encoder(img, ext, cv2.IMWRITE_JPEG_QUALITY,a=a)
        memo_cost += 1
        #memo_state = int(np.floor(memo_cost/max_memo)) # memo state becomes how much the memory limitation is exceeded
        #memo_state = 5 if memo_state > 5 else memo_state
        '''
        if memo_cost > max_memo:
            memo_state = 1 # exceed max memory
        else:
            memo_state = 0
        '''
    else:
        s['memo'] = s['memo'] # if don't record, the memory_state maintain

    # freq_state = int(np.floor(freq_ctr[WA_state_idx]/freq_threshold)) # frequency state becomes how frequent the state is
    # freq_state = 5 if freq_state > 5 else freq_state
    '''
    if freq_ctr[WA_state_idx] > freq_threshold:
        freq_state = 1 # the new WA_state is too frequent
    else:
        freq_state = 0
    '''
    '''Assign possible next state'''
    possible_state_list = []
    for idx in possible_state_idx:
        possible_s = s
        possible_s['i'] = idx
        state_array = compute_WA_state_array(idx)
        possible_s['u'] = compute_ambient_state_u(state_array)
        possible_s['freq'][idx] += 1
        possible_state_list.append(possible_s)

    # total_R = s[4] + s[0]*a[0]
    return possible_state_list, probabilities, memo_cost # 0 is normal, 1 is anomalous

def compute_reward(new_s, s, a):
    # we want the sequence have higher mean anomaly score, but we also want a higher total anomaly score.
    # we want to record more anomalous events
    # we would like to record a long sequence for each event, but use less memory.
    R_anomaly = compute_warning_anomaly_reward(new_s,s)
    R_recording = compute_recording_reward(new_s,s)
    R_memory = compute_memo_reward(new_s, s)
    R_freq = compute_freq_reward(new_s, s)

    reward = w[0]*R_anomaly + w[1]*R_recording + w[2]*R_memory + w[3]*R_freq
    return float(reward)

def compute_recording_reward(new_s,s):
    if new_s['rec'] >= 1:
        R_recording = -1 # negative reward if recording
    elif s['rec'] >= 1 and new_s['rec'] == 0:
        R_recording = 1  # positive reward if stop recording
    else:
        R_recording = 0
    return R_recording

def compute_warning_anomaly_reward(new_s,s):
    if new_s['rec'] == 1:
        R_anomaly = new_s['u']  # positive anomaly reward if start recording
    else:
        R_anomaly = 0
    return R_anomaly

def compute_memo_reward(new_s,s):
    if new_s['memo'] > max_memo:
        if new_s['rec'] == 0: # no record
            R_memory = 0
        elif new_s['rec']  == 1: # strong compress:
            R_memory = - 1
        elif new_s['rec'] == 2: # weak compress
            R_memory = -2
        else: # record w/o compress
            R_memory = -3
    else:
        R_memory = 0
    return R_memory

def compute_freq_reward(new_s,s):
    if new_s['rec'] == 1 and s['freq'][s['i']] > freq_threshold:
        R_freq = - 1
    else:
        R_freq = 0
    return R_freq

def decision_tree(s,i,memo_cost,img):
    inner_score_list = np.zeros(len(action_list)) # there are multiple possible new states given state and action
    outer_score_list = []
    for j,a in enumerate(action_list):
        new_s, probs, memo_cost = state_update(s, a, memo_cost, img) # should output list of possible new_s and their probs
        tmp = np.zeros(len(action_list)**(horizon-i))
        for k, possible_s in enumerate(new_s):
            new_r = compute_reward(possible_s, s, a)
            inner_score_list[j] += probs[k]*new_r
            # pdb.set_trace()
            if i < horizon:
                # outer_score_list = np.append(outer_score_list, new_r + _lambda* decision_tree(possible_s,i+1))
                tmp += probs[k]*(new_r + _lambda* decision_tree(possible_s,i+1, memo_cost,img))
        outer_score_list = np.append(outer_score_list, tmp)
    if i < horizon:
        return outer_score_list.astype(float)
    return inner_score_list.astype(float)

def compute_ambient_state_u(anomaly_scores):
    return np.sum(anomaly_scores[0:3]) + 0.5*np.sum(anomaly_scores[3:7])

if __name__ == "__main__":
    # initialize compressor
    compressor = simpleCompress()
    # read video
    video_path = '../Smart_Black_Box/data/videos/'
    video_name = '05182017_video1080p.mp4'
    cap = cv2.VideoCapture(video_path+video_name)
    frame_ctr = 0
    frame_time = 0.0333666


    buf_size = 0
    min_buf_size = 20
    s, U = init_state()
    file_name = '05182017.csv'
    anomaly_score, time = process_warning_anomaly(file_name)
    num_frames = anomaly_score.shape[0]
    # anomaly_score = [0,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0]
    prev_s = s
    file = open('optimal_action.txt', 'w')
    optimal_action_path = []
    i = 0
    while cap.isOpened():
        ret,img = cap.read()
        frame_ctr += 1
        '''Exit if the Mobileye data is finished'''
        if i >= num_frames:
            break
        '''downsample video(29.97fps) to data frame frequency(~10Hz)'''
        video_time = frame_ctr * frame_time  # + 2.8819 # the video is 30fps
        if video_time < time[i] - frame_time:
            # if video lag, keep reading video frames
            continue
        while video_time > time[i] + 2 * frame_time:
            # if message lag, keep reading message
            i = i + 1

        print ('Iteration: ', i)
        # s[0:num_WA] = anomaly_score[i,:]
        s['i'] = compute_WA_state_index(anomaly_score[i,:])
        s['u'] = compute_ambient_state_u(anomaly_score[i,:])
        s['freq'][s['i']] += 1

        # reward = compute_reward(s, prev_s)

        print ("State: ", s)
        # Q_value_list = reward + _lambda * decision_tree(s,1, memory_cost)
        Q_value_list = _lambda * decision_tree(s,1, memory_cost,img)

        print ("Q_value_list: ", Q_value_list)

        '''Find the optimal action'''
        max_Q = np.max(Q_value_list)
        max_idx = np.argmax(Q_value_list)
        optimal_action = action_list[int(max_idx/len(action_list)**(horizon-1))]

        '''The action should be recording if it's in recording mode and buffer size is smaller than the minimum size'''
        if s['rec']==1 and buf_size < min_buf_size:
            optimal_action = 1
        optimal_action_path.append(optimal_action)
        print ("Optimal action: ", optimal_action)

        '''Update state using the optimal action'''
        print("Cost of memory:", memory_cost)
        prev_s = s
        possible_s,_ ,memory_cost = state_update(s,optimal_action,memory_cost, img)
        s = possible_s[0] # since the system state portion is deterministic given action, we can select any possible state as the new state.
        if s['rec'] >= 1:
            buf_size += 1
        else:
            buf_size = 0

        i += 1
            # new_s = state_update(s, a)
            # r = compute_reward(new_s)
            # tmp_U = r +
        file.write("%s\n" % optimal_action)
    print ("Recording:", np.where(optimal_action_path!=0)[0])
    print ("Memory: ", s['memo'])
