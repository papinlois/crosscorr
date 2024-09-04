"""
Change of the usual picker code used on Tim's detections but this time for 
Bostock catalog: getting P and S times on stacked families.

- P and S
- stack families
- offset by family

NB: Always follow ### for personal comments + made for Talapas use.

@author: papin
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)
import obspy
from obspy import UTCDateTime
import glob
import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import unet_tools
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf') # using non-interactive backend

# ================ Functions ================

# Function to load and process daily .mseed data
def data_process(filePath, sampl=100):
    '''
    Load and process daily .mseed data.
    
    Args:
    - filePath: Absolute path of .mseed file.
    - sampl: Sampling rate.

    Returns:
    - D: Processed data object.
    '''
    if sampl == 100: # Tim's
        if not os.path.exists(filePath):
            return None
        D = obspy.read(filePath)
        if len(D) != 1:
            D.merge(method=1, interpolation_samples=-1, fill_value='interpolate')
        t1 = UTCDateTime(filePath.split('/')[-1].split('.')[0])
        t2 = t1 + 86400
        D.detrend('linear')
        D.taper(0.02) # 2% taper
        D.filter('highpass', freq=1.0)
        D.trim(starttime=t1 - 1, endtime=t2 + 1, nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1, method='linear')
        D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    return D

# Function to convert ZEN data to input feature
def ZEN2inp(Z, E, N, epsilon):
    '''
    Convert raw ZEN data to input feature.
    
    Args:
    - Z, E, N: Components of ZEN data.
    - epsilon: Small value for numerical stability.

    Returns:
    - data_inp: Processed input features.
    '''
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z) + epsilon)
    data_E_val = np.log(np.abs(E) + epsilon)
    data_N_val = np.log(np.abs(N) + epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1, 1), data_Z_sign.reshape(-1, 1),
                          data_E_val.reshape(-1, 1), data_E_sign.reshape(-1, 1),
                          data_N_val.reshape(-1, 1), data_N_sign.reshape(-1, 1)])
    return data_inp

# Function to perform quality control on data
def QC(data, Type='data'):
    '''
    Quality control of data.
    
    Args:
    - data: Input data to be checked.
    - Type: Type of data ('data' or 'noise').

    Returns:
    - bool: True if data passes all checks, False otherwise.
    '''
    if np.isnan(data).any():
        return False
    if np.max(np.abs(data)) == 0:
        return False
    data = data / np.max(np.abs(data))
    if Type == 'data':
        N1, N2, min_std, CC = 30, 30, 0.01, 0.98
    else:
        N1, N2, min_std, CC = 30, 30, 0.05, 0.98
    wind = len(data) // N1
    for i in range(N1):
        if np.std(data[int(i * wind):int((i + 1) * wind)]) < min_std:
            return False
    wind = len(data) // N2
    for i in range(N2):
        data_small = data[int(i * wind):int((i + 1) * wind)]
        data_bef = data[:int(i * wind)]
        data_aft = data[int((i + 1) * wind):]
        data_whole = np.concatenate([data_bef, data_aft])
        curr_CC = cal_CCC(data_whole, data_small)
        if curr_CC > CC:
            return False
    return True

# Function to calculate cross-correlation coefficient
def cal_CCC(data1, data2):
    '''
    Cross-correlation coefficient calculation.

    Args:
    - data1, data2: Input data arrays.

    Returns:
    - float: Maximum cross-correlation coefficient.
    '''
    from obspy.signal.cross_correlation import correlate_template
    CCCF = correlate_template(data1, data2)
    return np.max(np.abs(CCCF))

def calculate_seconds(dt):
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def plot_stack(stacked_traces, tr_id, family, detections):
    pairs = ['N', 'E', 'Z']
    # Plot each stack with an offset on the y-axis
    plt.figure(figsize=(12,6))
    nb = 1  # Distance between plots
    offset = len(stacked_traces) * nb
    x = np.linspace(0, 15, len(stacked_traces[0, :]), endpoint=False)
    for i in range(len(stacked_traces)):
        norm = np.max(np.abs(stacked_traces[i,:])) # Same weight for each stack on the figure
        plt.plot(x, stacked_traces[i,:]/norm+offset, label=f'{pairs[i]}')
        offset -= nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Stacked Traces for Family {family} '
               f'{tr_id} - {len(detections)} detections', fontsize=16)
    plt.yticks(np.arange(len(pairs))*nb+nb, pairs[::-1], fontsize=11)
    plt.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    # plt.xticks([0,5,10,20,30],[-5,0,5,15,25])
    plt.xlim(0,15)
    plt.ylim(0, len(pairs)*nb+nb)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/Stack_{family}_{tr_id}.png')
    plt.close()

# ================ Events ================

# Load station information ###old way but working so okay
phases = np.load('sav_family_phases.npy', allow_pickle=True).item()
all_sta = []
for k in phases.keys():
    all_sta += list(phases[k]['sta'].keys())
all_sta = list(set(all_sta))
filtered_sta = [station for station in all_sta if station in ### test
                {'SSIB', 'SILB', 'TSJB', 'TWKB', 'KLNB', 'LZB', 'PGC', 'PFB', 'MGCB'}]
print('Number of stations:', len(filtered_sta))

# Define all directories and mappings in a consolidated dictionary
network_directories = {
    'PO': {
        'dir': '/projects/amt/shared/cascadia_PO',
        'channels': {'default': 'HH'}
    },
    'CN': {
        'dir': '/projects/amt/shared/cascadia_CN',
        'channels': {
            'GOBB': 'EH',
            'LZB': 'BH',
            'MGB': 'EH',
            'NLLB': 'BH',
            'PFB': 'HH',
            'PGC': 'BH',
            'SNB': 'BH',
            'VGZ': 'BH',
            'YOUB': 'HH',
        }
    },
    'C8': {
        'dir': '/projects/amt/shared/cascadia_C8',
        'channels': {
            'BPCB': 'HH',
            'GLBC': 'HH',
            'JRBC': 'HH',
            'LCBC': 'HH',
            'MGCB': 'HH',
            'PHYB': 'HH',
            'SHDB': 'HH',
            'SHVB': 'HH',
            'SOKB': 'HH',
            'TWBB': 'HH',
        }
    }
}

# Read event detections and template dictionary ###lfe_svi
# startdate = datetime.strptime("20050903", "%Y%m%d")
# enddate = datetime.strptime("20050925", "%Y%m%d")
events = pd.read_csv('lfe_svi.txt', index_col=0, dtype={'date': str, 'hour': str, 'lfe_family' : str})
events['date'] = '20' + events['date']
events['date'] = pd.to_datetime(events['date'], format='%Y%m%d')
events['hour'] = events['hour'].str.zfill(2)
events['OT'] = events['date'] + pd.to_timedelta(events['hour'].astype(int), unit='h') + pd.to_timedelta(events['second'], unit='s')
# events = events[(events['OT'] >= startdate) & (events['OT'] < enddate)]
events = events.drop(columns=['hour','second'])
events = events.sort_values(by='OT', ascending=True)
events.reset_index(inplace=True)
events.index.name = 'Index'
print(events)

nb_family = events['lfe_family'].unique()
nb_family.sort()
print(nb_family)

# Details on the events
first_event = events.iloc[0]
print(f"\nFirst Event:\nOrigin Time (OT): {first_event['OT']}\nMagnitude (Mw): {first_event['Mw']}")
last_event = events.iloc[-1]
print(f"Last Event:\nOrigin Time (OT): {last_event['OT']}\nMagnitude (Mw): {last_event['Mw']}")
num_lfe_families = events['lfe_family'].nunique()
print(f"Number of LFE Families: {num_lfe_families}\n")
nb_events = events['lfe_family'].value_counts()
print(f"Here's the number of detections by family: {nb_events}")

# ================ Main Execution ================

def detc_sta_wave(sta):
    # Determine stations
    if sta in network_directories['CN']['channels']:
        network_info = network_directories['CN']
        net = 'CN'
    elif sta in network_directories['C8']['channels']:
        network_info = network_directories['C8']
        net = 'C8'
    else:
        network_info = network_directories['PO']
        net = 'PO'
    tar_dir = network_info['dir']
    chn = network_info['channels'].get(sta, 'HH')
    loc = ''
    tr_id = '.'.join([net, sta, chn])

    # Get daily data files ###
    print(f'Searching for data in: {tar_dir}/*.{net}.{sta}.{loc}.{chn}Z.mseed')
    D_Zs = glob.glob(f'{tar_dir}/*.{net}.{sta}.{loc}.{chn}Z.mseed')
    D_Zs.sort()
    print(f'Found {len(D_Zs)} data files')
    if len(D_Zs) == 0:
        return  # No data found for this station input

    # Create output file if it doesn't exist
    file_csv_P = f'./Detections_P_stack/cut_daily_{net}.{sta}.csv'
    if not os.path.exists(file_csv_P):
        with open(file_csv_P, 'w') as OUT1:
            OUT1.write('lfe_family,P_times,nb_events,y,id\n')
    file_csv_S = f'./Detections_S_stack/cut_daily_{net}.{sta}.csv'
    if not os.path.exists(file_csv_S):
        with open(file_csv_S, 'w') as OUT1:
            OUT1.write('lfe_family,S_times,nb_events,y,id\n')

    # Common parameters for stacking
    wid_sec = 15
    wid_pts = wid_sec * sr
    wid_T = np.arange(wid_pts) * (1.0 / sr)

    for family in nb_family:
        # Initialization
        data_Z = np.zeros(wid_pts)
        data_E = np.zeros(wid_pts)
        data_N = np.zeros(wid_pts)
        sav_data = []
        sav_data_Z = []
        sav_data_E = []
        sav_data_N = []
        cpt=0 #how many detections are not in the stack

        # Parameters of the stack
        events_family = events[events['lfe_family'] == family]
        events_family['seconds'] = events_family['OT'].apply(calculate_seconds)
        events_family['i_st'] = events_family['seconds'] * sr
        events_family['date'] = events_family['date'].dt.strftime('%Y%m%d')
        detections = events_family#.sort_values(by='Mw',ascending=False)['Mw'][:500]
        print(f'Stack of {len(detections)} events for family {family}:')
        print(detections.to_string())
        offset = np.floor(templates_dict_bostock[family]['min_p_wave'] * 2) / 2 * sr
        # offset = np.floor(float(templates_dict_bostock_stas[family][sta]) * 2) / 2 * sr ### per station
        print(offset)
        
        # Stacking detections
        for _, row in detections.iterrows():
            # Get the stream for the day of the detection
            date = row['date']
            D_Z = next((path for path in D_Zs if path.split('/')[-1][:8] == date), None)
            if D_Z:
                with open(file_csv_P, 'a') as OUT1: #P ou S doesn't matter
                    comp = D_Z.split('/')[-1].split('.')[-2]
                    D_E = D_Z.replace(comp, comp[:2] + 'E')
                    D_N = D_Z.replace(comp, comp[:2] + 'N')

                    # Check if the E, N components exist
                    if not os.path.exists(D_E) or not os.path.exists(D_N):
                        print(f'Missing at least one component! Data: {D_E} or {D_N} does not exist!')
                        continue

                    # Process the 3C data: detrend and high-pass filter
                    try:
                        D_Z = data_process(D_Z, sampl=sr)
                        D_E = data_process(D_E, sampl=sr)
                        D_N = data_process(D_N, sampl=sr)
                    except:
                        continue
    
                    assert len(D_Z[0].data) == len(D_E[0].data) == len(D_N[0].data) == 8640001, "Check data_process!"

                    # Extract numpy arrays
                    D_Z = D_Z[0].data
                    D_E = D_E[0].data
                    D_N = D_N[0].data
    
                    # Stack a number of detections for data
                    ## i_st is the index for OT, have to add the offset per family
                    i_st = int(row['i_st'])+int(offset)
                    i_ed = int(i_st + wid_pts)
                    data_Z += D_Z[i_st:i_ed]
                    data_E += D_E[i_st:i_ed]
                    data_N += D_N[i_st:i_ed]
                
            else:
                cpt+=1
                
        norm_val = max(max(np.abs(data_Z)), max(np.abs(data_E)), max(np.abs(data_N)))
        data_inp = ZEN2inp(data_Z / norm_val, data_E / norm_val, data_N / norm_val, epsilon)
        sav_data.append(data_inp)
        sav_data_Z.append(data_Z)
        sav_data_E.append(data_E)
        sav_data_N.append(data_N)
        sav_data = np.array(sav_data)
        
        # Plot les families stacked
        stacked_traces = np.vstack([data_N/norm_val, data_E/norm_val, data_Z/norm_val])        
        plot_stack(stacked_traces, tr_id, family, detections)
        
        ## Predict P wave times
        tmp_y_P = model_P.predict(sav_data) ### P&S
        idx_lfe_P = np.where(tmp_y_P.max(axis=1) >= thres)[0] ### P&S
        
        # Process each detected event
        for i_lfe in idx_lfe_P:
            D_merge = np.hstack([sav_data_Z[i_lfe], sav_data_E[i_lfe], sav_data_N[i_lfe]])
            D_merge = D_merge / np.max(np.abs(D_merge))
            if not QC(D_merge):
                continue  # Quality check failed, reject event
            idx_maxy_P = np.where(tmp_y_P[i_lfe] == np.max(tmp_y_P[i_lfe]))[0][0]
            wave_time_P = wid_T[idx_maxy_P] ### P&S
            with open(file_csv_P, 'a') as OUT1: ### P&S
                OUT1.write(f'{family},{wave_time_P:.2f},{len(detections)-cpt},{tmp_y_P[i_lfe][idx_maxy_P]:.2f},{tr_id}\n')

        ## Predict S wave times
        tmp_y_S = model_S.predict(sav_data) ### P&S
        idx_lfe_S = np.where(tmp_y_S.max(axis=1) >= thres)[0] ### P&S

        # Process each detected event
        for i_lfe in idx_lfe_S:
            D_merge = np.hstack([sav_data_Z[i_lfe], sav_data_E[i_lfe], sav_data_N[i_lfe]])
            D_merge = D_merge / np.max(np.abs(D_merge))
            if not QC(D_merge):
                continue  # Quality check failed, reject event
            idx_maxy_S = np.where(tmp_y_S[i_lfe] == np.max(tmp_y_S[i_lfe]))[0][0]
            wave_time_S = wid_T[idx_maxy_S] ### P&S
            with open(file_csv_S, 'a') as OUT1: ### P&S
                OUT1.write(f'{family},{wave_time_S:.2f},{len(detections)-cpt},{tmp_y_S[i_lfe][idx_maxy_S]:.2f},{tr_id}\n')

with open("templates_dict_bostock.json", 'r') as json_file:
    templates_dict_bostock = json.load(json_file)

# Parameters
all_sta = filtered_sta
n_cores = len(all_sta)
drop = False  # Flag for whether to use dropout in model
N_size = 2  # Size parameter for model
sr = 100
epsilon = 1e-6
thres = 0.1  # Threshold for detection

# Build model architecture
if drop:
    model_P = unet_tools.make_large_unet_drop(N_size, sr=sr, ncomps=3)
    model_S = unet_tools.make_large_unet_drop(N_size, sr=sr, ncomps=3)
else:
    model_P = unet_tools.make_large_unet(N_size, sr=sr, ncomps=3)
    model_S = unet_tools.make_large_unet(N_size, sr=sr, ncomps=3)

# Load weights from the latest checkpoint
run_num_P = 'P003'
run_num_S = 'S003'
chks_P = glob.glob(f"/projects/amt/shared/Vancouver_ML_LFE/checks/large*{run_num_P}*")
chks_S = glob.glob(f"/projects/amt/shared/Vancouver_ML_LFE/checks/large*{run_num_S}*")
chks_P.sort()
chks_S.sort()
model_P.load_weights(chks_P[-1].replace('.index', '')) ### P&S
model_S.load_weights(chks_S[-1].replace('.index', ''))

results = Parallel(n_jobs=-1, verbose=10)(delayed(detc_sta_wave)(sta) for sta in all_sta)
