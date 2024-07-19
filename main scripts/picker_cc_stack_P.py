"""
Code adapted from daily2input_S_20240326.py to run on days of the SSE 2005 for 
the PO network. It was originally for PB so comments can be the old version.

The picker stacks a number of the highest crosscorr coeff and search the AT of
S-wave on it. The code is written for every templates, and choosen numbers of
detections for stacks.

TODO:
    1. Faire same structure avec picker_tools et network_config.

NB: Always follow ### for personal comments + made for Talapas use.

@author: papin
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)
import obspy
from obspy import UTCDateTime
import glob
import os
# import sys
# import h5py
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import unet_tools
import matplotlib
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

# ================ Main Execution ================

# Load station information
phases = np.load('sav_family_phases.npy', allow_pickle=True).item()
all_sta = []
for k in phases.keys():
    all_sta += list(phases[k]['sta'].keys())
all_sta = list(set(all_sta))
filtered_sta = [station for station in all_sta if station in
                {'SSIB', 'SILB', 'TSJB', 'TWKB', 'KLNB'}]
print('Number of stations:', len(filtered_sta))

# Load station location info
stainfo = np.load('stainfo.npy', allow_pickle=True).item()

# Check if station coordinates are complete
for s in filtered_sta:
    if s not in stainfo:
        print(f"Station {s} not found in station information.")

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

# Read event detections and template dictionary
events = pd.read_csv('output_aug_PO_yes.txt')
with open('templates_dict.json') as f:
    templates_dict = json.load(f)

# Choose a template
# templ = '8'
# offset = templates_dict[templ]['offset']
# events = events[events['template'] == int(templ)]
# dates = events['starttime'].apply(lambda x: x[:10].replace('-', ''))
# dates = dates.unique()
# events['starttime'] = pd.to_datetime(events['starttime'])
# events['date_only'] = events['starttime'].dt.strftime('%Y%m%d')

# How many events
nb_events = events['template'].value_counts()
print(f"Here's the number of detections by template: {nb_events}")

# Parameters
run_num = 'P003'
drop = False  # Flag for whether to use dropout in model
N_size = 2  # Size parameter for model
sr = 100
epsilon = 1e-6
thres = 0.1  # Threshold for detection

# Build model architecture
if drop:
    model = unet_tools.make_large_unet_drop(N_size, sr=100, ncomps=3)
else:
    model = unet_tools.make_large_unet(N_size, sr=100, ncomps=3)

# Load weights from the latest checkpoint
chks = glob.glob("/projects/amt/shared/Vancouver_ML_LFE/checks/large*%s*" % (run_num))
chks.sort()
model.load_weights(chks[-1].replace('.index', ''))

def detc_sta(sta):
    # How many detections stacked
    nb_stack=[5,2,1]#450,350,300,250,200,150,100,50,20,10,
    for nb in nb_stack:
        # For all templates from the catalog
        nb_templ = events['template'].unique()
        for templ in nb_templ:
            # Select the events of the template
            events_templ=events[events['template'] == int(templ)]
            dates = events_templ['starttime'].apply(lambda x: x[:10].replace('-', ''))
            dates = dates.unique()
            events_templ['starttime'] = pd.to_datetime(events_templ['starttime'])
            events_templ['date_only'] = events_templ['starttime'].dt.strftime('%Y%m%d')

            # Verify if enough events are present to stack
            if len(events_templ)>=nb:

                # Determine stations
                if sta in network_directories['CN']['channels']:
                    network_info = network_directories['CN']
                elif sta in network_directories['C8']['channels']:
                    network_info = network_directories['C8']
                else:
                    network_info = network_directories['PO']

                net = sta if sta in network_directories else 'PO'  # Default to PO if not found
                tar_dir = network_info['dir']
                chn = network_info['channels'].get(sta, 'HH')
                loc = ''

                # Get daily data files
                print('Searching for data in:', tar_dir+'/200509*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
                D_Zs = glob.glob(tar_dir+'/200509*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
                D_Zs.sort()
                print('Found %d data files' % (len(D_Zs)))
                if len(D_Zs) == 0:
                    return  # No data found for this station input

                # Create output files
                file_csv = './Detections_P_PO_stack/cut_daily_%s.%s.csv' % (net, sta)
                # file_hdf5 = './Detections_S_PO_stack/cut_daily_%s.%s.hdf5' % (net, sta)
                # Create CSV file
                if not os.path.exists(file_csv):
                    OUT1 = open(file_csv,'w')
                    OUT1.write('nb_stack,mean_cc,lfe_family,P_times,y,idx_max,id\n')
                    OUT1.close()
                # else:
                #     print("File: cut_daily_all.csv already exist! Exit and not overwritting everything")
                #     sys.exit()

                # # Create HDF5 file
                # hf = h5py.File(file_hdf5, 'a')
                # hf.create_group('data')
                # hf.close()

                # Get locations
                stlon, stlat, stdep = stainfo[sta]

                # Determine event times
                detections = events_templ.sort_values(by='coeff', ascending=False)
                detections['seconds'] = detections['starttime'].apply(calculate_seconds)
                detections['i_st'] = detections['seconds'] * sr
                detections['coeff']=events_templ['coeff']

                # Cut daily data into 15-second segments
                wid_sec = 15
                wid_pts = wid_sec * sr
                wid_T = np.arange(wid_pts) * (1.0 / sr)
                data_Z = np.zeros(wid_pts)
                data_E = np.zeros(wid_pts)
                data_N = np.zeros(wid_pts)
                sav_data = []
                sav_data_Z = []
                sav_data_E = []
                sav_data_N = []

                # Parameters of the stack
                detections = detections.head(nb)
                print(detections)
                mean_cc = np.mean(detections['coeff'])

                for _, row in detections.iterrows():
                    date = row['date_only']
                    D_Z =  next((path for path in D_Zs if path.split('/')[-1][:8] == date), None)
                    OUT1 = open(file_csv, 'a')
                    # hf = h5py.File(file_hdf5, 'a')
                    comp = D_Z.split('/')[-1].split('.')[-2]
                    D_E = D_Z.replace(comp, comp[:2]+'E')
                    D_N = D_Z.replace(comp, comp[:2]+'N')

                    # Check if the E, N components exist
                    if (not os.path.exists(D_E)) or (not os.path.exists(D_N)):
                        print('Missing at least one component! Data:', D_E, 'or', D_N, 'does not exist!')
                        # hf.close()
                        OUT1.close()
                        continue

                    # Process the 3C data: detrend and high-pass filter
                    try:
                        D_Z = data_process(D_Z, sampl=sr)
                        D_E = data_process(D_E, sampl=sr)
                        D_N = data_process(D_N, sampl=sr)
                    except:
                        # hf.close()
                        OUT1.close()
                        continue

                    assert len(D_Z[0].data) == len(D_E[0].data) == len(D_N[0].data) == 8640001, "Check data_process!"

                    # Extract numpy arrays
                    D_Z = D_Z[0].data
                    D_E = D_E[0].data
                    D_N = D_N[0].data

                    # Stack a number of detections for data
                    i_st = int(row['i_st'])
                    i_ed = int(i_st + wid_pts)
                    data_Z += D_Z[i_st:i_ed]
                    data_E += D_E[i_st:i_ed]
                    data_N += D_N[i_st:i_ed]

                norm_val = max(max(np.abs(data_Z)), max(np.abs(data_E)), max(np.abs(data_N)))
                data_inp = ZEN2inp(data_Z / norm_val, data_E / norm_val, data_N / norm_val, epsilon)
                sav_data.append(data_inp)
                sav_data_Z.append(data_Z)
                sav_data_E.append(data_E)
                sav_data_N.append(data_N)

                sav_data = np.array(sav_data)
                tmp_y = model.predict(sav_data)
                idx_lfe = np.where(tmp_y.max(axis=1) >= thres)[0]

                # Process each detected event
                for i_lfe in idx_lfe:
                    D_merge = np.hstack([sav_data_Z[i_lfe], sav_data_E[i_lfe], sav_data_N[i_lfe]])
                    D_merge = D_merge / np.max(np.abs(D_merge))
                    if not QC(D_merge):
                        continue  # Quality check failed, reject event
                    idx_maxy = np.where(tmp_y[i_lfe] == np.max(tmp_y[i_lfe]))[0][0]
                    P_time=wid_T[idx_maxy]
                    tr_id = '.'.join([net, sta, chn])
                    OUT1.write('%d,%.2f,%s,%.2f,%.2f,%d,%s\n' %
                               (nb, mean_cc, templ, P_time, tmp_y[i_lfe][idx_maxy],
                                idx_maxy, tr_id))

                OUT1.close()

# Parallel processing
all_sta = filtered_sta
print(all_sta)
n_cores = len(all_sta)
results = Parallel(n_jobs=n_cores, verbose=10)(delayed(detc_sta)(sta) for sta in all_sta)
