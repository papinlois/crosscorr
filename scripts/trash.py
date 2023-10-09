# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:26:35 2023

@author: papin
"""

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
