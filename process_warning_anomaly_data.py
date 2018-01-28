from data_reader import *
import matplotlib.pyplot as plt
import seaborn as sns

def load_warning_anomaly(file_name):
    warning_path = 'data/warnings/'
    # warning_name = '05182017.csv'
    warnings = read_data(warning_path+file_name)
    warnings = warnings[28:27655+28,:]

    anomaly_path = 'data/anomalies/'
    # anomaly_name = '05182017.csv'
    anomalies = read_data(anomaly_path+file_name)

    return warnings, anomalies

def process_warning_anomaly(file_name):
    warnings, anomalies = load_warning_anomaly(file_name)
    '''
    Read and process the warning signals
    '''
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

    '''
    Process computed anomaly scores
    '''
    thresh1 = 0.6
    thresh2 = 0.2
    time = anomalies[:,0]
    Rt = anomalies[:,1] / np.max(anomalies[:,1])
    Rd = anomalies[:,2] / np.max(anomalies[:,2])
    Rb = anomalies[:,3] / np.max(anomalies[:,3])
    Ro = anomalies[:,4] / np.max(anomalies[:,4])

    Rt[Rt>thresh1] = 2
    Rt[Rt<thresh2] = 0
    Rt[np.array(Rt>thresh2) & np.array(Rt<thresh1)] = 1

    Rd[Rd>thresh1] = 2
    Rd[Rd<thresh2] = 0
    Rd[np.array(Rd>thresh2) & np.array(Rd<thresh1)] = 1

    Rb[Rb>thresh1] = 2
    Rb[Rb<thresh2] = 0
    Rb[np.array(Rb>thresh2) & np.array(Rb<thresh1)] = 1

    Ro[Ro>thresh1] = 2
    Ro[Ro<thresh2] = 0
    Ro[np.array(Ro>thresh2) & np.array(Ro<thresh1)] = 1

    warning_anomaly = np.vstack([FCW, LDW, FSW, Rt, Rd, Rb, Ro]) # remove FSW since it was a constant in 05182017 data
    warning_anomaly = warning_anomaly.T

    return warning_anomaly.astype(int), time