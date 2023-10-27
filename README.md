# Cross-Correlation and Template Matching with ObsPy

This is a Python script for performing cross-correlation and template matching with ObsPy on seismic data. The script takes a set of seismic traces and a collection of seismic templates, and it calculates cross-correlations to detect seismic events. The script also generates various plots and outputs relevant information.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Script Overview](#script-overview)
- [Output Files](#output-files)
- [Author](#author)

## Introduction

This script is designed to analyze seismic data and detect seismic events by performing cross-correlation between the data and a collection of seismic templates. It utilizes ObsPy, a Python toolbox for seismology, to process and analyze seismic waveforms.

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

## Script Overview

1. **Data Ingestion**: It reads seismic data from the specified files, ensuring you have the right seismic dataset at your disposal.

2. **Data Preprocessing**: The script prepares the data by interpolating, trimming, detrending, and filtering. 

3. **Template Matching**: Seismic templates are loaded, and the script iterates through these templates and stations to calculate cross-correlations. 

4. **Significant Event Detection**: The script identifies significant correlations and generates plots to visualize these events. 

5. **Output Generation**: The script produces informative files and reports containing data about the detected events and the script's execution time.

## Output Files

The script generates a variety of output files, including:

- `threshold.txt`: This file contains threshold and maximum cross-correlation values, offering insights into the significance of the detected events.

- `info.txt`: An information-rich report that encompasses essential details about the script's execution, the date range analyzed, the stations and channels used, templates, and non-significant correlations.

- Correlation plots: Visual representations of significant events, helping you visualize the correlation between templates and seismic data.

## Author

This script was crafted by [papin](https://github.com/papin), and it is made available as an open-source tool to advance seismic data analysis. Feel free to adapt and enhance the script to meet your specific research needs. If you encounter questions or require assistance, please don't hesitate to reach out to the author through their GitHub profile for guidance and support.
