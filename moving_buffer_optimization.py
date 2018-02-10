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


'''Run moving horizon optimization'''
if __name__ == '__main__':
    # initialize compressor
    output_path = 'recorded_img/05182017/'
    compressor = simpleCompress(output_path)
    # read video
    video_path = '../Smart_Black_Box/data/videos/'
    video_name = '05182017_video1080p.mp4'
    cap = cv2.VideoCapture(video_path+video_name)
    frame_ctr = 0
    frame_time = 0.0333666


    # file_name = '05182017.csv'
    # anomaly_score, time_array = process_warning_anomaly(file_name)
    # num_frames = anomaly_score.shape[0]
    three_warnings, states_list, value_list, time_array = process_data()
    '''Use statics from a larger data set'''
    states_list = np.array([[0,0,0],
                            [0,0,1],
                            [0,1,0],
                            [1,0,0]])
    value_list = np.array([0.04319829,5.37383929,7.80472283,10.10175974])
    print(states_list)
    print(value_list)
    num_frames = three_warnings.shape[0]
    file = open(output_path + 'optimal_action.txt', 'w')
    optimal_action_path = []
    i = 0

    '''Parameters and buffers'''
    img_buf = []
    buf_size = 500
    overlap = 50
    eta = 5
    zeta = 10
    sigma = 10
    memo_max = 200000
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
    test_data = three_warnings[0:27655, :]  # [6500:8500,:]#[47000:47500,:]#[15200:15400,:]
    num_data = test_data.shape[0]

    print("Data reading succeeded!")
    input("continue...")

    # pool = mp.Pool(processes=4)
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
        # print(img_size)
        # input("continue")
        moving_buf['size'].append(img_size/100)  # assume the img size follows a normal distribution
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
            memo_tracker += float(np.dot(sorted_action, np.array(moving_buf['size'])))

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
                        compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3,persistent_record=True)
                else:
                    if policy > 0:
                        compressor.run_opencv(img_buf[j], '.jpeg', cv2.IMWRITE_JPEG_QUALITY, quality=100, i=i-buf_size+j, j=j, a=3, persistent_record=True)
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

    file = open(output_path + 'optimal_action.txt', 'w')
    for policy in optimal_policy:
        file.write("%s\n" % policy)
    file.close()

    plt.figure(1)
    plt.plot(test_data[:, 0], 'r')
    plt.plot(test_data[:, 1], 'g')
    plt.plot(test_data[:, 2], 'b')
    plt.ylim([-1, 2])
    plt.legend(['FCW', 'LDW', 'FSW'])

    plt.figure(2)
    plt.subplot(211)
    plt.plot(all_raw_scores)
    plt.plot(all_filtered_scores)
    plt.legend(['Raw data value signal', 'Filtered signal'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig('data_value_signal.png',dpi=500)

    # plt.figure(3)
    plt.subplot(212)
    plt.plot(optimal_policy, 'k')
    plt.legend(['Optimal policy path'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=2)
    # plt.savefig('value_and_policy_MIQP.png',dpi=500)
