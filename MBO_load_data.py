from data_reader import *

def process_data():
    '''Read data'''
    warning_path = 'data/warnings/'
    warning_name = '05182017.csv'
    warnings_05182017 = read_data(warning_path+warning_name)
    warnings_05182017 = warnings_05182017[28:27655+28,:]
    print(warnings_05182017.shape)

    # warning_name = '10152017.csv'
    # warnings_10152017 = read_data(warning_path+warning_name)
    # print(warnings_10152017.shape)
    #
    # warning_name = '10172017.csv'
    # warnings_10172017 = read_data(warning_path+warning_name)
    # print(warnings_10172017.shape)
    #
    # warning_name = '10172017_freeway.csv'
    # warnings_10172017_freeway = read_data(warning_path+warning_name)
    # print(warnings_10172017_freeway.shape)

    anomaly_path = 'data/anomalies/'
    anomaly_name = '05182017.csv'
    anomalies_05182017 = read_data(anomaly_path+anomaly_name)
    print(anomalies_05182017.shape)
    #
    # anomaly_name = '10152017.csv'
    # anomalies_10152017 = read_data(anomaly_path+anomaly_name)
    # print(anomalies_10152017.shape)
    #
    # anomaly_name = '10172017.csv'
    # anomalies_10172017 = read_data(anomaly_path+anomaly_name)
    # print(anomalies_10172017.shape)
    #
    # anomaly_name = '10172017_freeway.csv'
    # anomalies_10172017_freeway = read_data(anomaly_path+anomaly_name)
    # print(anomalies_10172017_freeway.shape)

    '''concate data'''
    warnings = np.vstack([warnings_05182017])#, warnings_10152017, warnings_10172017, warnings_10172017_freeway])
    anomalies = np.vstack([anomalies_05182017])#, anomalies_10152017, anomalies_10172017, anomalies_10172017_freeway])
    print(warnings.shape)
    print(anomalies.shape)

    '''Read and process the warning signals'''
    FCW = warnings[:,1].astype(int)# Forward Collision Warning
    LLDW = warnings[:,2].astype(int) # Left Lane Departure Warning
    RLDW = warnings[:,3].astype(int) # Right Lane Departure Warning
    LDW = LLDW|RLDW
    FSW = warnings[:,4].astype(int) # FailSafe Warning
    FSType = warnings[:,5].astype(int) # FailSafe Types

    LCrossing = warnings[:,7]
    RCrossing = warnings[:,8]
    LCutIn = warnings[:,9]
    RCutIn = warnings[:,10]

    '''Compute probabilities of Warning variables, as value of data'''
    three_warnings = np.hstack([np.reshape(FCW,(len(FCW),1)),np.reshape(LDW,(len(LDW),1)),np.reshape(FSW,(len(FSW),1))])
    states_list,state_counts = np.unique(three_warnings,axis=0,return_counts=True)
    print (states_list)
    print (state_counts)
    p_list = state_counts/three_warnings.shape[0]
    print (p_list)
    value_list = -np.log2(p_list)

    time = anomalies[:, 0]

    return three_warnings, states_list, value_list, time
