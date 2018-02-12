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

'''Utilities'''
def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        # field = [field_name]
        writer = csv.writer(csvfile)
        for i in range(data.shape[0]):
            writer.writerow(data[i,:])

def gaussian_func(x,a,mu,sigma):
    return a*np.exp(-((x-mu)**2)/(2*sigma**2))
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




def run_MBO(cap,test_data,
            states_list,value_list,
            time_array,
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
    buf_size = 500
    overlap = 50

    sigma = 10
    memo_max = 200000000
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

    img_size_list = []
    while cap.isOpened():
        start_time = time.time()

        ret,img = cap.read()
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
        img_size = compressor.run_opencv(img, '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=0, j=0, a=3, persistent_record=False)
        img_size_list.append(img_size)
        # print(img_size)
        # input("continue")
        moving_buf['size'].append(img_size/100)  # rescale the image size
        img_buf.append(img)

        if len(moving_buf['value']) == buf_size or i == num_data - 1:  # start to optimize if buffer is full or there is no new frame
            buf_length = len(moving_buf['value'])
            '''Filter the score'''
            filtered_score = score_filter(np.array(moving_buf['value']), sigma)

            '''Init the MIQP problem'''
            prob = pic.Problem()
            pi = [prob.add_variable(str(j), 1, vtype='binary') for j in range(buf_length)]
            size = pic.new_param('size', moving_buf['size'])
            value = pic.new_param('value', filtered_score)

            '''Add objective function'''
            obj_func = (size[0] - eta * value[0]) * pi[0]
            for j in range(1, buf_length):
                obj_func += (size[j] - eta * value[j]) * pi[j] + zeta * (pi[j] - pi[j - 1]) ** 2
            prob.set_objective('min', obj_func)

            '''Add constraint'''
            prob.add_constraint(pic.sum([size[j] * pi[j] for j in range(buf_length)],  # summands
                                        'j',  # name of the index
                                        '[buf_length]'  # set to which the index belongs
                                        ) < np.max([memo_max - memo_tracker, 0])
                                )
            '''Solve'''
            sol = prob.solve(solver='gurobi', verbose=0)
            sorted_action = [float(policy.value[0]) for policy in pi]

            '''Save data, update storage capacity'''
            # memo_tracker += float(np.dot(sorted_action, np.array(moving_buf['size'])))

            '''Append the sorted action to the whole action list'''
            if k == 0:
                optimal_policy = sorted_action
                all_filtered_scores = filtered_score
                all_raw_scores = np.array(moving_buf['value'])
            else:
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


            '''Write images to hard disk'''
            for j,policy in enumerate(sorted_action):
                if len(overlap_policy) > 0 and j < overlap:
                    if overlap_policy[j] > 0: # if the data has been recorded, don't record again
                        continue
                    elif overlap_policy[j] == 0 and policy > 0:
                        written_size = compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3,persistent_record=False)
                        memo_tracker += written_size
                else:
                    if policy > 0:
                        written_size = compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3, persistent_record=False)
                        memo_tracker += written_size

            print("=======================")
            print("Buffer number: ", k)
            print("Local optimal recording: ", np.where(np.array(sorted_action)>0)[0])
            #input("continue...")
            # start the next buffer
            k += 1
            overlap_policy = sorted_action[buf_size - overlap:]
            overlap_score = moving_buf['value'][buf_size-overlap:]
            overlap_filtered_score = filtered_score[buf_size-overlap:]

            moving_buf['size'] = moving_buf['size'][buf_size-overlap:]
            moving_buf['value'] = moving_buf['value'][buf_size-overlap:]
            moving_buf['state'] = moving_buf['state'][buf_size-overlap:]
            img_buf = img_buf[buf_size-overlap:]

        i += 1
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
    output_path = 'recorded_img/05182017/'
    # file = open(output_path + 'optimal_action.txt', 'w')

    compressor = simpleCompress(output_path)
    # read video
    video_path = '../Smart_Black_Box/data/videos/'
    video_name = '05182017_video1080p.mp4'

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

    eta_list = [1]#[1,2,3,4,5,6,7,8,9,10]
    zeta_list = [1]#[1,2,3,4,5,6,7,8,9,10]
    anomaly_memory_ratio_matrix = np.zeros([len(eta_list), len(zeta_list)])
    event_memory_ratio_matrix = np.zeros([len(eta_list), len(zeta_list)])
    min_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])
    max_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])
    mean_event_length_matrix = np.zeros([len(eta_list), len(zeta_list)])

    for i, eta in enumerate(eta_list):
        for j, zeta in enumerate(zeta_list):

            cap = cv2.VideoCapture(video_path + video_name)
            '''Run MBO'''
            optimal_policy, total_memory_cost, img_size_list = run_MBO(cap, test_data, states_list, value_list, time_array, eta=eta, zeta=zeta)


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

    print (anomaly_memory_ratio_matrix)
    print (event_memory_ratio_matrix)
    print (min_event_length_matrix)
    print (max_event_length_matrix)
    print (mean_event_length_matrix)
    write_csv('img_size_05182017.csv', np.array(img_size_list))
    # write_csv('anomaly_memory_ratio.csv', anomaly_memory_ratio_matrix)
    # write_csv('event_memory_ratio.csv', event_memory_ratio_matrix)
    # write_csv('min_event_length.csv', min_event_length_matrix)
    # write_csv('max_event_length.csv', max_event_length_matrix)
    # write_csv('mean_event_length.csv', mean_event_length_matrix)


