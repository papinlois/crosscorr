# Cross-Correlation Analysis Tool

This Python script performs cross-correlation analysis on seismic data collected from different stations and channels. It computes cross-correlations between pairs of stations, detects significant correlations, and visualizes the results.

## Prerequisites

Before using this script, make sure you have the following prerequisites installed:

- Python 3.x
- ObsPy: A Python toolbox for seismology. Install it using `pip`:

  ```bash
  pip install obspy
  ```

- Other required Python libraries (NumPy, Pandas, Matplotlib, and SciPy) can be installed using `pip`:

  ```bash
  pip install numpy pandas matplotlib scipy
  ```

## Usage

1. Clone this repository to your local machine.

2. Create a CSV file named `stations.csv` containing station information. The file should have the following columns:

   - `Name`: Station name
   - `Longitude`: Longitude of the station
   - `Latitude`: Latitude of the station

   Example `stations.csv`:

   ```csv
   Name,Longitude,Latitude
   SNB,-120.123,36.789
   LZB,-121.456,35.678
   ```

3. Place your seismic data files (in MiniSEED format) in a directory. The script expects data files to be named following the pattern `YYYYMMDD.CN.STATION.CHANNEL.mseed`. For example:

   - `20100520.CN.SNB.BHE.mseed`
   - `20100520.CN.LZB.BHN.mseed`

4. Modify the script's configuration variables:

   - `stas`: List of station codes to analyze (e.g., `['SNB', 'LZB']`).
   - `channels`: List of channel codes to read (e.g., `['BHE', 'BHN', 'BHZ']`).
   - Adjust other parameters (e.g., filtering options) as needed.

5. Run the script:

   ```bash
   python cross_correlation_analysis.py
   ```

6. The script will process the data, perform cross-correlation analysis, and generate plots showing significant correlations between stations. Results will be displayed in the terminal and saved as plot files.

## Output

- The script will generate plots showing significant correlations between station pairs.
- If no significant correlations are found, it will display a message indicating so.

## Acknowledgments

This code uses the ObsPy library for seismological data analysis.
