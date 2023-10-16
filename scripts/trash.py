# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:26:35 2023

@author: papin
"""

# Cross-correlation for 1 set of data
for ii, tr in enumerate(st):
    xcorrfull = np.zeros((numwindows, tr.stats.npts - windowlen + 1))
    for kk in range(numwindows):
        xcorrfull[kk, :] = autocorr_tools.correlate_template(tr.data,tr.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)],
                                                              mode='valid', normalize='full', demean=True, method='auto')
    xcorrmean += xcorrfull

#############################################################################


#Use of the cross-correlation function of numpy for 2 signals

xcorr_result = np.correlate(tr1.data, tr2.data, mode='full')

# Time values for x-axis
sampling_rate = tr1.stats.sampling_rate
time_values = np.arange(-len(tr1.data) + 1, len(tr1.data)) / sampling_rate

# Plot cross-correlation result
plt.figure(figsize=(10, 6))
plt.plot(time_values, xcorr_result, color='b', linewidth=2)
plt.xlabel('Time Lag (s)', fontsize=14)
plt.ylabel('Cross-Correlation', fontsize=14)
plt.title('Cross-Correlation between SNB and LZB', fontsize=16)
plt.grid(True)

# Show the plot
plt.show()

#############################################################################

for kk in range(numwindows):
    cc1 = xcorrfull[kk, :] = autocorr_tools.correlate_template(tr1.data, tr2.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)],
                                                                mode='valid', normalize='full', demean=True, method='auto')
    cc2 = correlate_template(tr1.data, tr2.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)])

    # Print values when the subtraction is not zero
    non_zero_diff = cc1 - cc2
    non_zero_indices = np.nonzero(non_zero_diff)
    for idx in range(len(non_zero_indices[0])):
        print(f"Non-zero difference at index {non_zero_indices[0][idx]}: {non_zero_diff[non_zero_indices[0][idx]]}")

#############################################################################

# Calculate the number of rows and columns for subplots
num_plots = len(stas) * len(channels)
num_cols = int(math.ceil(num_plots / 3))  # You can change the number of columns as needed
num_rows = int(math.ceil(num_plots / num_cols))

# Set the figure size based on the number of subplots
fig_width = 15  # Adjust as needed
fig_height = 5 * num_rows  # Adjust as needed

# Create a figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

# Plot the data
offset = 0
for sta_idx, sta in enumerate(stas):
    for cha_idx, cha in enumerate(channels):
        ax = axes[offset // num_cols, offset % num_cols]  # Get the current subplot axis
        tr = st[sta_idx * len(channels) + cha_idx]
        shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
        color = (0, 0, 0.5 + shade / 2)
        ax.plot(tr.times("timestamp"), tr.data / np.max(np.abs(tr.data)) + offset, color=color, label=f"{sta}_{cha}")
        ax.set_xlabel('Timestamp', fontsize=14)
        ax.set_ylabel('Normalized Data + Offset', fontsize=14)
        ax.set_title(f"Station: {sta}, Channel: {cha}", fontsize=12)
        offset += 1

# Adjust the layout of subplots
plt.tight_layout()

# Save the figure
plt.savefig('C:/Users/papin/Desktop/phd/plots/data_plot.png')

# Close the figure
plt.close()

#############################################################################

import datetime

# Unix timestamps for the x-axis limits
start_unix_timestamp = 1274217000
end_unix_timestamp = 1274218000

# Convert Unix timestamps to datetime objects
start_datetime = datetime.datetime.utcfromtimestamp(start_unix_timestamp)
end_datetime = datetime.datetime.utcfromtimestamp(end_unix_timestamp)

# Print the corresponding timestamps in human-readable format
print("Start Time:", start_datetime)
print("End Time:", end_datetime)

#############################################################################

def find_threshold_multiplier(xcorrmean, threshold_values):
    # Calculate the MAD of xcorrmean
    mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
    
    # Initialize a dictionary to store results
    results = {}
    
    # Iterate through threshold values
    for multiplier in threshold_values:
        threshold = multiplier * mad
        aboves = np.where(xcorrmean > threshold)
        num_significant_correlations = aboves[0].size
        results[multiplier] = num_significant_correlations
    
    return results

# Example usage
threshold_values = [2, 3,4, 5,6, 7,8]  # Adjust these values as needed
threshold_results = find_threshold_multiplier(xcorrmean, threshold_values)

# Find the multiplier that gives the desired number of significant correlations
desired_num_significant = 1000  # Adjust this value as needed
best_multiplier = None
for multiplier, num_significant in threshold_results.items():
    if num_significant >= desired_num_significant:
        best_multiplier = multiplier
        break

print(f"Best multiplier for {desired_num_significant} significant correlations: {best_multiplier}")

#############################################################################

current_xlim = plt.xlim()
tick_positions, tick_labels = plt.xticks()
plt.xlim(1274217000,1274218000)

#############################################################################

# Add locations
for ii, sta in enumerate(stas):
    ind = np.where(locs[:, 0] == sta)
    st[ii].stats.y = locs[ind, 1][0][0]
    st[ii].stats.x = locs[ind, 2][0][0]
