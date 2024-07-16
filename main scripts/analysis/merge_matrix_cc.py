#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:57:04 2024

!! Nothing checks that idx1 and idx2 are the same when merging (supposed to 
be okay)

@author: lpapin
"""

import glob
import pandas as pd
base_dir = "/Users/lpapin/Desktop/SSE_2005/dendrogram/"
peaks=[1,2,3,4,5,6]
which=['xcorr','xcf']
for cc in which:
    # print(cc)
    for peak in peaks:
        # print(peak)
        outputs = glob.glob(base_dir + f'output_{cc}*')
        # print(outputs)
        for output in outputs:
            cc_values = pd.read_csv(output)
            if output == outputs[0]:
                matrix_cc = cc_values[['idx1', 'idx2', 'time_event1', 'time_event2']].copy()
            idx = peak + 1
            matrix_cc.loc[:, cc_values.columns[2]] = cc_values.iloc[:, idx]

        matrix_cc = matrix_cc.sort_values(by=['idx1', 'idx2'])
        output_file_path = base_dir + f"xcval_{cc}_peak{peak}.txt"
        matrix_cc.to_csv(output_file_path, sep='\t', index=False, float_format='%.3f')