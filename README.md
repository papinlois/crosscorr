# CrossCorr Seismic Data Analysis

This repository contains Python scripts for analyzing and detecting seismic events in seismic station data using network cross-correlation techniques. The scripts are designed to process and analyze seismic data from multiple stations, perform cross-correlation, and detect significant correlations, which may indicate seismic events. The detected events are then clustered and summarized for further analysis.

## Table of Contents
- [Getting Started](#getting-started)
- [Repository Overview](#repository-overview)
- [Script Overview](#script-overview)
- [Output](#output)
- [Author](#author)

## Getting Started

To use this seismic data analysis and detection script, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/papinlois/crosscorr.git
   ```

2. Navigate to the repository folder:

   ```bash
   cd crosscorr
   ```

3. Place your seismic data files in MiniSEED format in a directory of your choice. Update the `path` variable in the `crosscorr.py` script to point to the data directory. 

4. Modify the network parameters of the stations you want to use in your study in `network_configuration.py`:

   - `stations`: List of station codes to analyze (e.g., `['SNB', 'LZB']`). You can specify the stations you want to include in the analysis.

   - `channels`: List of channel codes to read (e.g., `['BHE', 'BHN', 'BHZ']`). Customize the channels to read from your seismic data.
 
   - `filename_pattern`: How is construct the filename of the streams you are going to read (e.g., `'{date}.CN.{station}..{channel}.mseed'`).

6. Adjust other parameters directly in `crosscorr.py` such as filtering options, network code, and the date of interest as needed to suit your specific data and analysis requirements.

### Running the Analysis

Run the main script, `crosscorr.py`, to perform seismic data analysis and detection:

```bash
python crosscorr.py
```

The script will process the seismic data, perform network cross-correlation, detect events, and create plots and output files with the results.

### View Results

The repository will generate various plots, output files, and information about detected seismic events in the specified date range and station data. You can find the results in the `plots` directory and the `output.txt` file.

## Repository Overview

- `crosscorr.py`: The main script for seismic data analysis and detection.
- `crosscorr_bostock.py`: This script is specified for a catalog of LFEs in Cascadia (Bostock et al., 2015).
- `crosscorr_tools.py` and `autocorr_tools.py`: Modules containing utility functions for data visualization and processing.
- `network_configurations.py`: A dictionary of all the stations with their channels and filename paths. Imported as a module in the main script of cross-correlation.
- `EQloc_001_0.1_3_S.csv`: Catalog of our choice for the template matching (used in `crosscorr.py`).
- `create_arrival_times.py` and `create_arrival_times_bostock.py`: Scripts that creates all P- and S-waves arrival times for the selected catalog (`arrival_times_tim.txt` and `arrival_times_bostock.txt`). It also plots different figures to have an idea of the times for selected stations.

## Script Overview

This repository contains Python scripts for seismic data analysis and detection using cross-correlation techniques. The scripts are designed to perform the following tasks:

1. **Data Ingestion**:
   - Reads seismic data from specified files, ensuring you have the right seismic dataset at your disposal.
2. **Data Preprocessing**:
   - Prepares the data by interpolating, trimming, detrending, and filtering for further analysis.
3. **Template Matching**:
   - Loads seismic templates and iterates through these templates and stations to calculate cross-correlations.
4. **Significant Event Detection**:
   - Identifies significant correlations and generates plots to visualize these events.
5. **Iterative Process**:
	- Stacks the detections for a higher SNR ratio, and continues the cross-correlation with it.
6. **Output Generation**:
   - Produces informative files and reports containing data about the detected events and the script's execution time.

## Output

### Plots

The repository generates a series of plots in the `plots` directory. These plots include:

-  `templ<templ_index>_template1_<lastday>.png`: Show of the detection and its window on every stations used for computation that are used as first template.
-  `templ<templ_index>_crosscorr<iteration>_<lastday>.png`: Cross-correlation plots highlighting the significant correlations. 
- `templ<templ_index>_stack<iteration>_<lastday>.png`: Stack of the new detections made for each station, that will be used as new templates in the process.

### `output.txt`

The `output.txt` file contains information about detected seismic events. Each line in the file provides the following details:

- `starttime`: The start time of the detected event.
- `template`: Template index associated with the event.
- `cc value`: The cross-correlation value for the detected event.
- `run`: The number of the iteration when the detection was made.

### `info.txt`

The `info.txt` file serves as a summary and information log for the seismic data analysis and detection process. It includes the following details:

- Date Range: The time period for which the analysis was performed, spanning from the start date to the end date.

- Stations and Channel Used: Lists the stations and channels selected for analysis.

- Templates: Provides information about the loaded seismic templates, including their datetime, latitude, longitude, and residual values.

## Author

This script was crafted by [papinlois](https://github.com/papinlois), and it is made available as an open-source tool to advance seismic data analysis. Feel free to adapt and enhance the script to meet your specific research needs. If you encounter questions or require assistance, please don't hesitate to reach out to the author through their GitHub profile for guidance and support.

As of 20/05/2024.

