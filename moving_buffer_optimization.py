from __future__ import division
import numpy as np
from data_reader import *
from MBO_load_data import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import copy
import cv2
import time
import picos as pic
import cvxopt as cvx
from compressor import simpleCompress
from data_reader import *
from heapq import *
import pdb
from scipy.optimize import minimize

memo_max = 500000
inflation_factor = 1.001


'''Utilities'''
def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        # field = [field_name]
        writer = csv.writer(csvfile)
        for i in range(data.shape[0]):
            writer.writerow([data[i]])

def gaussian_func(x,a,mu,sigma):
    return a*np.exp(-((x-mu)**2)/(2*sigma**2))

def comp_qual(qual):
    '''Given quality value, compute the corresponding compression ratio.''' 
    #compression_ratio = 0.05121*2.71812**(1.896*qual) + (4.352*10**(-11))*2.71828**(23.45*qual)
    #compression_ratio[np.where(compression_ratio > 1)] = 1
    #compression_ratio[np.where(compression_ratio < 0)] = 0

    compression_ratio = 0.10138*np.tan(1.4697608*qual)
    return compression_ratio

def score_filter(raw_score,sigma):
    j = 0
    N = raw_score.shape[0]
    filtered_score = copy.deepcopy(raw_score)
    for i,score in enumerate(raw_score):
        if j > 0:
            if raw_score[j] > raw_score[j-1]:
                mu = j
                start_idx = np.max([mu - 3*sigma,0])
                a = raw_score[j] - raw_score[j-1]
                for k in range(start_idx,mu):
                    filtered_score[k] = np.max([filtered_score[k],gaussian_func(k,a,mu,sigma)])
            elif raw_score[j] < raw_score[j-1]:
                mu = j
                end_idx = np.min([mu + 3*sigma,N])
                a = raw_score[j-1] - raw_score[j]
                for k in range(mu,end_idx):
                    filtered_score[k] = np.max([filtered_score[k],gaussian_func(k,a,mu,sigma)])
        j += 1
    return filtered_score

def compute_raw_score(states_list,state,value_list):
    tmp_i = np.where((states_list == state).all(axis=1))[0]
    return value_list[tmp_i]

# def supervisor(recoring_list,
#                 recording_frame_index_list,
#                 memory_list, value_list):
#     supervisor_policy = supervisor_optimization(recoring_list, memory_list, value_list)
#     memo_tracker = np.dot(memory_list, supervisor_policy)
#     remove_idx = np.where(supervisor_policy==0)[0]
#     for idx in remove_idx:
#         del memory_list[int(idx)]
#         del value_list[int(idx)]
#         del recording_frame_index_list[int(idx)]
#
#     return recording_frame_index_list, memory_list, value_list
#
# def supervisor_optimization(memory_list, value_list):
#     num_recordings = len(memory_list)
#     '''Init the MIQP problem'''
#     prob = pic.Problem()
#     pi = [prob.add_variable(str(j), 1, vtype='binary') for j in range(num_recordings)]
#     size = pic.new_param('size', memory_list]
#     value = pic.new_param('value', value_list)
#
#     '''Add objective function'''
#     obj_func = 0
#     for j in range(1, num_recordings):
#         obj_func += value_list[j]  * pi[j]
#     prob.set_objective('max', obj_func)
#
#     '''Add constraint'''
#     prob.add_constraint(pic.sum([size[j] * pi[j] for j in range(buf_length)],  # summands
#                                 'j',  # name of the index
#                                 '[buf_length]'  # set to which the index belongs
#                                 ) < memo_max
#                         )
#     '''Solve'''
#     sol = prob.solve(solver='gurobi', verbose=0)
#     sorted_action = [float(policy.value[0]) for policy in pi]
#     return sorted_action

def get_recording_info(tmp_recording,total_recording_number):
    max_value = 0
    total_cost = 0
    total_value = 0

    for frame_info in tmp_recording:
        if frame_info[0] > max_value:
            max_value = frame_info[0]
        total_cost += frame_info[2]
        total_value += frame_info[0]
    start_idx = tmp_recording[0][1]
    end_idx = tmp_recording[-1][1]
    inflated_value = max_value * inflation_factor**total_recording_number
    return (inflated_value,start_idx, end_idx,total_cost)

def add_recordings_to_heapq(recording_heapq,tmp_recording,
                            total_recording_number,i,
                            buf_decisions,
                            filtered_score,
                            moving_buf):
    curr_buf_size = len(buf_decisions)
    for j, decision in enumerate(buf_decisions):
        if decision > 0:
            prev_decision = decision
            tmp_recording.append(
                (float(filtered_score[j]), i-curr_buf_size+1+j, moving_buf['size'][j]*100))  # (value, index, cost) of each frame
        else:
            if len(tmp_recording) > 0:
                recording_info = get_recording_info(tmp_recording,total_recording_number)
                # check whether old recordings need to be removed from heapq
                heappush(recording_heapq,recording_info)
                total_cost_in_heapq = 0
                for recording in recording_heapq:
                    total_cost_in_heapq += recording[3]
                while total_cost_in_heapq >= memo_max: # check which recordings to remove
                    removed_recording = heappop(recording_heapq)
                    # if removed_recording[0] < recording_info[0]:
                    total_cost_in_heapq -= removed_recording[3]
                    print ("one recording is removed:", removed_recording)
                    # else:
                    #     heappush(recording_heapq,removed_recording)
                    #     break
                # heappush(recording_heapq, recording_info)
                tmp_recording = []  # reset
                total_recording_number += 1
                print ("num recordings:",total_recording_number)
            # if decision > 0:
            #     tmp_recording.append(
            #         (filtered_score[j], i, moving_buf['size'][j]))  # (value, index, cost) of each frame
            # else:
            #     if len(tmp_recording) > 0:
            #         recording_info = get_recording_info(tmp_recording)
            #         heappush(recording_heapq, recording_info)
            #         tmp_recording = []  # reset
        if j > buf_size - overlap and len(buf_decisions) == buf_size:
            break
    return recording_heapq, tmp_recording, total_recording_number

def picos_optimize(moving_buf, filtered_score, buf_length):
    '''Init the MIQP problem'''
    prob = pic.Problem()
    pi = [prob.add_variable(str(j), 1, vtype='continuous') for j in range(buf_length)]
    size = pic.new_param('size', moving_buf['size'])
    value = pic.new_param('value', filtered_score)

    '''Add objective function'''
    obj_func = (size[0] - eta * value[0]) * pi[0]
    for j in range(1, buf_length):
        # obj_func += (size[j] - eta * value[j]) * pi[j] + zeta * (pi[j] - pi[j - 1]) ** 2
        obj_func += size[j] * comp_qual(pi[j]) - eta * value[j] * pi[j] + zeta * (pi[j] - pi[j - 1]) ** 2
    prob.set_objective('min', obj_func)

    '''Add constraint'''
    # prob.add_constraint(pic.sum([size[j] * pi[j] for j in range(buf_length)],  # summands
    #                            'j',  # name of the index
    #                            '[buf_length]'  # set to which the index belongs
    #                            ) < np.max([memo_max*100 - memo_tracker, 0])
    #                    )
    # prob.add_constraint(pic.sum([size[j] * comp_qual(pi[j]) for j in range(buf_length)],  # summands
    #                            'j',  # name of the index
    #                            '[buf_length]'  # set to which the index belongs
    #                            ) < np.max([memo_max*100 - memo_tracker, 0])
    #                    )
    for j in range(buf_length):
        prob.add_constraint(pi[j] <= 1)
        prob.add_constraint(pi[j] >= 0)
    '''Solve'''
    sol = prob.solve(solver='gurobi', verbose=0)
    sorted_action = [float(policy.value[0]) for policy in pi]

    return sorted_action

def scipy_obj_func(x,size,value):
    term1 = np.dot(size, comp_qual(x)) # size term
    term2 = eta * np.dot(value,x) # value term
    term3 = 0 # continuity term
    for j in range(1,len(size)):
        term3 += (x[j] - x[j - 1]) ** 2
    term3 *= zeta
    return term1 - term2 + term3

def scipy_optimize(moving_buf, filtered_score, buf_length):
    size = np.reshape(moving_buf['size'],(buf_length,))
    value = np.reshape(filtered_score,(buf_length,))
    x0 = 0.5 * np.ones(buf_length)
    cons = ({'type': 'ineq',
             'fun': lambda x: np.array(1 - x)},
            {'type': 'ineq',
             'fun': lambda x: np.array(x)})
    res = minimize(scipy_obj_func, x0, args=(size,value), constraints=cons, method='SLSQP', options={'disp': True})

    action = res.x
    action[np.where(action < 0)] = 0
    # print("eta:", eta)
    # print("zeta:",zeta)
    #
    # print("size:", size)
    # print("value:", value)
    return action

def run_MBO(cap,test_data,
            states_list,value_list,
            time_array,img_size_list,
            eta = 5,zeta = 10):
    frame_ctr = 0
    frame_time = 0.0333666

    # file_name = '05182017.csv'
    # anomaly_score, time_array = process_warning_anomaly(file_name)
    # num_frames = anomaly_score.shape[0]

    optimal_action_path = []
    i = 0

    '''Parameters and buffers'''
    img_buf = []
    sigma = 10

    moving_buf = {}
    moving_buf['size'] = []
    moving_buf['value'] = []
    moving_buf['state'] = []
    optimal_policy = []
    overlap_policy = [] # save the end of the last buffer.
    k = 0 # count the number of buffer
    memo_tracker = 0
    all_filtered_scores = []
    all_raw_scores = []

    num_data = test_data.shape[0]
    # pool = mp.Pool(processes=4)
    start_time = 0
    # img_size_list = []
    recording_heapq = [] # use heapq to save recordings
    total_recording_number = 0
    # memory_list = []
    # value_list = []

    while cap.isOpened():

        # ret,img = cap.read()
        frame_ctr += 1
        '''Exit if the Mobileye data is finished'''
        if i >= num_data:
            break
        '''downsample video(29.97fps) to data frame frequency(~10Hz)'''
        video_time = frame_ctr * frame_time  # + 2.8819 # the video is 30fps

        if video_time < time_array[i] - frame_time:
            # if video lag, keep reading video frames
            # if s['rec'] >= 1:
            #     compressor.run_opencv(img, ext, cv2.IMWRITE_JPEG_QUALITY, i=i,j=frame_ctr, a=s['rec'], persistent_record=True)
            continue

        while video_time > time_array[i] + 2 * frame_time:
            # if message lag, keep reading message
            i += 1

        '''Buffer append'''
        moving_buf['state'].append(test_data[i, :])
        moving_buf['value'].append(compute_raw_score(states_list, test_data[i, :], value_list))
        # img_size = compressor.run_opencv(img, '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=0, j=0, a=3, persistent_record=False)
        img_size = float(img_size_list[i])
        # img_size_list.append(img_size)
        moving_buf['size'].append(img_size/100)  # rescale the image size
        # img_buf.append(img)
        if len(moving_buf['value']) == buf_size or i == num_data - 1:  # start to optimize if buffer is full or there is no new frame
            # print("Time elapse:",time.time() - start_time)
            # start_time = time.time()
            buf_length = len(moving_buf['value'])
            '''Filter the score'''
            filtered_score = score_filter(np.array(moving_buf['value']), sigma)

            '''Run optimization'''
            # sorted_action = picos_optimize(moving_buf, filtered_score, buf_length)
            sorted_action = scipy_optimize(moving_buf, filtered_score, buf_length)
            '''Save data, update storage capacity'''
            # memo_tracker += float(np.dot(sorted_action, np.array(moving_buf['size'])))
            '''Append the sorted action to the whole action list'''
            if k == 0:
                optimal_policy = sorted_action
                all_filtered_scores = filtered_score
                all_raw_scores = np.array(moving_buf['value'])

                # size, value and frame indeces for
                tmp_recording = [] # used to save (value, index, cost) of each frame in a recording
                recording_heapq,tmp_recording,total_recording_number = \
                    add_recordings_to_heapq(
                        recording_heapq,
                        tmp_recording,
                        total_recording_number,
                        i,
                        optimal_policy,
                        filtered_score,
                        moving_buf)

            else:
                current_buf_size = len(sorted_action)
                # tmp_recording = []
                # update the overlapped policy and score if the newly computed are better
                tmp_end = len(optimal_policy)
                for j in range(overlap):
                    if optimal_policy[tmp_end-overlap+j] < sorted_action[j]:
                        optimal_policy[tmp_end - overlap + j] = sorted_action[j]
                    if all_filtered_scores[tmp_end-overlap+j] < filtered_score[j]:
                        all_filtered_scores[tmp_end - overlap + j] = filtered_score[j]
                optimal_policy = np.append(optimal_policy, sorted_action[overlap:])
                all_filtered_scores = np.append(all_filtered_scores, filtered_score[overlap:])
                all_raw_scores = np.append(all_raw_scores, np.array(moving_buf['value'][overlap:]))

                '''following part is used to save recordings to heapq'''

                buf_decisions = optimal_policy[-current_buf_size:]
                recording_heapq, tmp_recording, total_recording_number = \
                    add_recordings_to_heapq(
                        recording_heapq,
                        tmp_recording,
                        total_recording_number,
                        i,
                        buf_decisions,
                        filtered_score,
                        moving_buf)
                # for j,decision in enumerate(buf_decisions):
                #     if j == 0:
                #         if decision > 0:
                #             prev_decision = decision
                #             tmp_recording.append(
                #                 (filtered_score[j], i, moving_buf['size'][j]))  # (value, index, cost) of each frame
                #         else:
                #             if len(tmp_recording) > 0:
                #                 recording_info = get_recording_info(tmp_recording)
                #                 heappush(recording_heapq, recording_info)
                #                 tmp_recording = []  # reset
                #     else:
                #         if decision > 0:
                #             tmp_recording.append(
                #                 (filtered_score[j], i, moving_buf['size'][j]))  # (value, index, cost) of each frame
                #         else:
                #             if len(tmp_recording) > 0:
                #                 recording_info = get_recording_info(tmp_recording)
                #                 heappush(recording_heapq, recording_info)
                #                 tmp_recording = []  # reset
                #     if j >buf_size-overlap:
                #         break

            '''Write images to hard disk'''
            for j,policy in enumerate(sorted_action):
                if len(overlap_policy) > 0 and j < overlap:
                    if overlap_policy[j] >= policy: # if the data has been recorded with higher quality, don't record again
                        continue
                    else: # otherwise, remove the old low quality recording and add high quality recording.
                        # written_size = compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3,persistent_record=False)
                        old_size = img_size_list[i-buf_size+j] * overlap_policy[j]
                        written_size = img_size_list[i-buf_size+j] * policy
                        memo_tracker -= old_size
                        memo_tracker += written_size
                else:
                    if policy > 0:
                        #written_size = compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3, persistent_record=False)
                        written_size = img_size_list[i-buf_size+j] * policy
                        memo_tracker += written_size

            print("=======================")
            print("Buffer number: ", k)
            #print("Local optimal recording: ", np.where(np.array(sorted_action)>0)[0])
            print("Local optimal recording: ", sorted_action)
            #input("continue...")
            # start the next buffer
            k += 1
            overlap_policy = sorted_action[buf_size - overlap:]
            overlap_score = moving_buf['value'][buf_size-overlap:]
            overlap_filtered_score = filtered_score[buf_size-overlap:]

            moving_buf['size'] = moving_buf['size'][buf_size-overlap:]
            moving_buf['value'] = moving_buf['value'][buf_size-overlap:]
            moving_buf['state'] = moving_buf['state'][buf_size-overlap:]
            # img_buf = img_buf[buf_size-overlap:]
        i += 1
    print(recording_heapq)
    return optimal_policy, memo_tracker,img_size_list
    # file = open(output_path + 'optimal_action.txt', 'w')
    # for policy in optimal_policy:
    #     file.write("%s\n" % policy)
    # file.close()
    #
    # plt.figure(1)
    # plt.plot(test_data[:, 0], 'r')
    # plt.plot(test_data[:, 1], 'g')
    # plt.plot(test_data[:, 2], 'b')
    # plt.ylim([-1, 2])
    # plt.legend(['FCW', 'LDW', 'FSW'])
    #
    # plt.figure(2)
    # plt.subplot(211)
    # plt.plot(all_raw_scores)
    # plt.plot(all_filtered_scores)
    # plt.legend(['Raw data value signal', 'Filtered signal'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, mode="expand", borderaxespad=0.)
    # # plt.savefig('data_value_signal.png',dpi=500)
    #
    # # plt.figure(3)
    # plt.subplot(212)
    # plt.plot(optimal_policy, 'k')
    # plt.legend(['Optimal policy path'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.tight_layout(pad=2)
    # # plt.savefig('value_and_policy_MIQP.png',dpi=500)

'''Run moving buffer optimization'''
if __name__ == '__main__':
    buf_size = 500
    overlap = 50

    output_path = 'recorded_img/05182017/'
    # file = open(output_path + 'optimal_action.txt', 'w')

    compressor = simpleCompress(output_path)
    # read video
    video_path = '../Smart_Black_Box/data/videos/'
    video_name = '05182017_video1080p.mp4'
    # load image size from csv file to save running time
    img_size_list = read_data('data/img_size_05182017.csv')

    three_warnings, states_list, value_list, time_array = process_data()
    test_data = three_warnings[0:27655, :]  # [6500:8500,:]#[47000:47500,:]#[15200:15400,:]

    '''Use statics from a larger data set'''
    states_list = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])
    value_list = np.array([0.04319829, 5.37383929, 7.80472283, 10.10175974])
    print(states_list)
    print(value_list)
    print("Data reading succeeded!")
    input("continue...")

    eta_list = [5]#[1,2,3,4,5,6,7,8,9,10]
    zeta_list = [10]#[1,2,3,4,5,6,7,8,9,10]
    anomaly_memory_ratio_matrix = np.zeros([len(eta_list), len(zeta_list)])
    event_memory_ratio_matrix = np.zeros([len(eta_list), len(zeta_list)])
    min_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])
    max_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])
    mean_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])
    number_of_events_matrix = np.zeros([len(eta_list), len(zeta_list)])
    number_of_frames_matrix = np.zeros([len(eta_list), len(zeta_list)])


    for i, eta in enumerate(eta_list):
        for j, zeta in enumerate(zeta_list):
            cap = cv2.VideoCapture(video_path + video_name)
            '''Run MBO'''
            optimal_policy, total_memory_cost, img_size_list = run_MBO(cap, test_data, states_list, value_list, time_array, img_size_list,eta=eta, zeta=zeta)

            '''Compute and print result'''
            total_recorded = len(np.where(optimal_policy > 0)[0])
            recorded_anomalies= 0
            recorded_events = 0
            # recorded_anomalies_event = 0
            event_flag = False
            event_length = 0
            event_length_list = []
            for k, frame in enumerate(test_data):
                if np.sum(frame) > 0 and optimal_policy[k] > 0:
                    recorded_anomalies += 1
                if optimal_policy[k] > 0 and optimal_policy[k-1] == 0:
                    recorded_events += 1
                    event_flag = True
                if event_flag == True:
                    event_length += 1
                    if optimal_policy[k] == 0:
                        event_flag = False
                        event_length_list.append(event_length)
                        event_length = 0

            if total_memory_cost == 0:
                anomaly_memory_ratio = 0
                event_memory_ratio = 0
            else:
                anomaly_memory_ratio = recorded_anomalies/(total_memory_cost/1024)
                event_memory_ratio = recorded_events/(total_memory_cost/1024)

            if len(event_length_list) == 0:
                min_event_length = 0
                max_event_length = 0
                mean_event_length = 0
            else:
                min_event_length = np.min(event_length_list)
                max_event_length = np.max(event_length_list)
                mean_event_length = np.mean(event_length_list)
            print ("Number of total recorded frames: " + str(total_recorded))
            print ("NUmber of total recorded events:" + str(recorded_events))
            print ("Total memory cost: " + str(total_memory_cost/1024) + " MB   /   " + str(total_memory_cost/1024**2) + " GB")
            print ("Event length list: ", event_length_list)
            print ("The anomaly/memory ratio is: " + str(anomaly_memory_ratio) + " frame/MB")
            print ("The event/memory ratio is: " + str(event_memory_ratio) + " event/MB")
            print ("The min event length is: " + str(min_event_length))
            print ("The max event length is: " + str(max_event_length))
            print ("The mean event length is: " + str(mean_event_length))

            anomaly_memory_ratio_matrix[i,j] = anomaly_memory_ratio
            event_memory_ratio_matrix[i,j] = event_memory_ratio
            min_event_length_matrix[i,j] = min_event_length
            max_event_length_matrix[i,j] = max_event_length
            mean_event_length_matrix[i,j] = mean_event_length
            number_of_events_matrix[i,j] = recorded_events
            number_of_frames_matrix[i,j] = total_recorded

    print (anomaly_memory_ratio_matrix)
    print (event_memory_ratio_matrix)
    print (min_event_length_matrix)
    print (max_event_length_matrix)
    print (mean_event_length_matrix)
    print (number_of_events_matrix)
    print (number_of_frames_matrix)
    # write_csv('img_size_05182017.csv', np.array(img_size_list))
    # write_csv('anomaly_memory_ratio.csv', anomaly_memory_ratio_matrix)
    # write_csv('event_memory_ratio.csv', event_memory_ratio_matrix)
    # write_csv('min_event_length.csv', min_event_length_matrix)
    # write_csv('max_event_length.csv', max_event_length_matrix)
    # write_csv('mean_event_length.csv', mean_event_length_matrix)
