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

3. Place your seismic data files in MiniSEED format in a directory of your choice. Update the `path` variable in the `crosscorr_talapas.py` script to point to the data directory. 

4. Modify the network parameters of the stations you want to use in your study in `network_configurations_talapas.py`.

5. Adjust other parameters directly in `crosscorr_talapas.py` such as filtering options, network code, and the date of interest as needed to suit your specific data and analysis requirements.

### Running the Analysis

Run the main script, `crosscorr_talapas.py`, to perform seismic data analysis and detection:

```bash
python crosscorr_talapas.py
```

The script will process the seismic data, perform network cross-correlation, detect events, and create plots and output files with the results.

### View Results

The repository will generate various plots, output files, and information about detected seismic events in the specified date range and station data. You can find the results in the `plots` directory and the `output.txt` file.

## Repository Overview

### Main scripts

- `crosscorr_talapas.py`: The main script for seismic data analysis and detection. It is aimed to be used on the cluster (because of the filepaths and such).
- `crosscorr_tools.py` and `autocorr_tools.py`: These are modules containing utility functions for data visualization and processing.
- `network_configurations_talapas.py`: A dictionary of all the stations with their channels and filename paths. Imported as a module in the main script of cross-correlation.
- `matric_cc`: This script computes all the cross-correlation values between unique events pairs and creates .txt files with all the results. Our choice here is our new events detected by the template-matching.
- `picker_cc_stack.py`: Any number of detections can be stacked and we use a picker to get the S-wave arrival time.

### Analysis

All scripts in the analysis folder contribute to the analysis of either the data before to computation or the outcomes of the many main scripts. Some of the more useful ones are:

- `catalog_stats.py`: Statistics of the catalog used in the template-matching.
- `create_arrival_times.py`: Script that creates all P- and S-waves arrival times for the selected catalog. It also plots different figures to have an idea of the times for selected stations. *Last update:* `crosscorr_tools` has a version of this code as a function, called in `crosscorr_talapas.py`.
- `events_study.py`: Plots that illustrate the seismic events choosen.
- `output_stats.py`: Statistics of the detections made with the template-matching.
- `picks_stats.py`: Plots the different S-wave arrival times computed for different number of detections stacked. This allows to compare how the time evolves with the quality of the stack.
- `plot_data2.py`: Plots the streams of any choosen stations and dates.
- `plot_detections.py`: This scripts plot the original event and the detections with the highest cross-correlation coefficients. Also plot the S-wave arrival time picked if it exists.

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

**NB:** Something to keep in mind when running the code is that each situation is unique, and it is essential to plan ahead of time which networks and stations will be used. It may be interesting to limit ourselves to only one network with that code structure for the time being because we are computing network cross-correlation (SNR presumably more homogeneous).

## Output

### Plots

The repository generates a series of plots in the `plots` directory. These plots include:

-  `templ<templ_index>_AT_<subfolder>.png`: Show the expected arrival times of P- and S-waves for every stations. Also plots the interval that will be used as a the window of the template in the cross-correlation, defined by the first P-wave arrival time.
-  `templ<templ_index>_template1_<day>.png`: Show the detection and its window on every stations used for computation that are used as first template.
-  `templ<templ_index>_crosscorr<iteration>_<day>.png`: Cross-correlation plots highlighting the significant correlations (=detections). 
- `templ<templ_index>_stack<iteration>_<day>.png`: Stack of the new detections made for each station, that will be used as new templates in the process.

### `output.txt`

The `output.txt` file contains information about detected seismic events. Each line in the file provides the following details:

- `starttime`: The start time of the detected event.
- `template`: Template index associated with the event.
- `coeff`: The cross-correlation value for the detected event.
- `run`: The number of the iteration when the detection was made.

### `info.txt`

The `info.txt` file serves as a summary and information log for the seismic data analysis and detection process. It includes the following details:

- Date Range: The time period for which the analysis was performed, spanning from the start date to the end date.

- Stations and Channel Used: Lists the stations and channels selected for analysis.

- Templates: Provides information about the loaded seismic templates, including their origin time, coordinates, and number of stations that detected the event. Also gives the index of the templates that didn't satisfy the requirement of detections.

## Author

This script was crafted by [papinlois](https://github.com/papinlois), and it is made available as an open-source tool to advance seismic data analysis. Feel free to adapt and enhance the script to meet your specific research needs. If you encounter questions or require assistance, please don't hesitate to reach out to the author through their GitHub profile for guidance and support.

As of 10/07/2024.

