# Cross-Correlation Analysis for Seismic Data

This repository contains a Python script for performing cross-correlation analysis on seismic data from multiple stations. The script reads seismic data files, preprocesses the data, calculates cross-correlations between different station pairs, and identifies significant correlations.

## Overview

The script provided in this repository performs the following tasks:

1. Load seismic data from specified stations and channels.
2. Preprocess the data, including interpolation, trimming, detrending, and filtering.
3. Calculate cross-correlations between different station pairs.
4. Identify significant correlations based on a threshold (8 times the Median Absolute Deviation).
5. Save detection plots and correlation function plots for further analysis.

## Usage

To use the script, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/your-username/your-repository.git
   ```

2. Install the required dependencies. You can install ObsPy using the following command:

   ```shell
   pip install obspy
   ```

3. Create a `stations.csv` file with station information, including station names, longitudes, and latitudes.

4. Place your seismic data files (in MiniSEED format) in a directory and update the `path` variable in the script to point to the data directory.

5. Modify the script's configuration variables:

   - `stas`: List of station codes to analyze (e.g., `['SNB', 'LZB']`).
   - `channels`: List of channel codes to read (e.g., `['BHE', 'BHN', 'BHZ']`).
   - Adjust other parameters (e.g., filtering options) as needed.

6. Run the script:

   ```shell
   python crosscorr.py
   ```

7. The script will perform the analysis and generate detection plots and correlation function plots.

8. Results, including significant correlation values, will be saved in a `results.txt` file.

## Acknowledgments

This code uses the ObsPy library for seismological data analysis.
