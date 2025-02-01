# **FPL Optimiser**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [*1. Configuration*](#1-configuration)
  - [*2. Data preparation*](#2-data-preparation)
  - [*3. Running the optimiser*](#3-running-the-optimiser)
  - [*4. Output*](#4-output)
- [Contributing](#contributing)

## Overview

**FPL Optimiser** is a Python tool designed to assist managers in making data-driven decisions reagrding optimal team selection in the [Fantasy Premier League (FPL)](https://fantasy.premierleague.com/) football game. The optimiser formulates and solves an [**integer linear programming** (ILP)](https://en.wikipedia.org/wiki/Integer_programming) problem, where forecasted points over a specified gameweek interval are maximised subject to a set of constraints based on FPL game rules. These constraints include player positions, budget limits, and team selection rules, all of which are critical to ensuring the team's validity in the game. 

This tool not only enables the optimisation of team selection for upcoming gameweeks using forecast data but also provides the ability to calculate optimal team selections for historic gameweeks using actual data observed in previous gameweeks. The combination of forecast-based and actual data-driven optimisation makes FPL Optimiser a versatile tool for FPL managers aiming for the best possible team composition.

## Key Features

- **Multi-Gameweek Optimisation**: Supports optimisation over multiple gameweeks to account for longer-term planning and strategy.
- **Real-Time Calculation**: Performs efficient calculations to deliver an optimal team selection in real time, providing users with actionable insights immediately.
- **Use of Existing Team**: Users can specify an existing team as a starting point for the optimisation process, allowing them to forecast and optimize their current squad rather than starting from scratch.
- **Transfer Support**: The optimiser incorporates the simulation of transfers for team optimisation, allowing users to make decisions regarding player acquisitions and removals.
- **User-Friendly Integration**: Designed for both technical users and FPL enthusiasts with an easy-to-understand input/output structure for efficient usage.
- **Reporting**: Clear and consise outputs are reported to the user covering: optimal team for the gameweek, additional player details, expected points, and other relevant metrics. 

## Project Structure

- **`fpl_optimiser/`**: Main project directory.
  - **`data/`**: Sub-directory containing historical gameweek and forecast data, modules also provided to manipulate and process data.
    - **`actuals/`**: Historical gameweek data folder.
    - **`fpl_xpts_forecast_data/`**: Gameweek points forecast data folder.
    - **`official_api_data/`**: Official FPL API gamweek data folder.
    - `data.py`: Module containing classes that enable key data processing. 
    - `enrichment.py/`: Module containing data enrichment logic.
  - **`optimiser/`**: Sub-directory containing optimiser classes.
    - `optimiser.py`: Module specifying the future forecast gameweek optimiser class.
    - `optimiser_actuals.py`: Module specifying the historic gameweek optimiser class.
  - **`utils/`**: Contains utility functions and classes
    - `constants.py`: Module specifying key project variables. 
    - `yaml_loader.py`: Module implementing YAML file utility routines.
- **`config/`**: Holds configuration files like YAML files for project settings and paths.
- **`scripts/`**: Contains scripts covering showing example usage and gameweek forecasting.
- **`.gitignore`**: Lists the files and directories that should be excluded from version control.
- **`README.md`**: Provides documentation for the project.
- **`requirements.txt`**: Specifies the required Python packages for the project.

## Dependencies

The following Python libraries are defined in requirements.txt:

- `pandas==2.2.2`: Used for data manipulation and analysis.
- `numpy==2.0.2`: Essential for numerical operations and array manipulations.
- `PuLP==2.9.0`: A linear programming library used to solve the integer linear programming (ILP) problems for team optimisation.
- `requests==2.32.3`: Used for making HTTP requests, including API calls to retrieve data from external sources.
- `PyYAML==6.0.2`: A library for parsing and working with YAML configuration files.

## Installation

To get started with the FPL Optimiser, follow the steps below:

### 1. Clone the Repository

Clone the repository to your local machine:
```bash
git clone https://github.com/Rohan-Gill/fpl_optimiser.git
```

### 2. Install dependencies

Navigate to the project directory and install the required dependencies using pip:
```bash
cd fpl_optimiser
pip install -r requirements.txt
```

## Usage

### 1. Configuration

The optimiser uses a configuration file, create a copy of the distributed YAML file (config_dist.YAML) and rename to config.YAML.

Make sure this file is correctly configured according to your needs:
```bash
season: "2024/2025"  # Update the season

fpl_api_base_url: "https://fantasy.premierleague.com/api"  # Base URL for FPL API

FPL_TEAM_ID: 1234567  # Update to your FPL team ID

# Authentication cookie for accessing the FPL API.
fpl_api_cookie_auth: |-
  "Your authentication cookie"  # Update with your cookie
```

### 2. Data preparation

The optimiser requires player gameweek points projection data. This can be input to the solver as a Pandas DataFrame object or can be sourced from the following sub-directories:

- **`data/fpl_xpts_forecast_data/`**
- **`data/official_api_data/`**
- **`data/actuals/`**

Ensure that the appropriate data files are present in these directories before running the optimiser. If using your own data, ensure the format and structure of the data is aligned to that obtained in section 3 (see below).

### 3. Running the optimiser

To run the optimiser one must instantiate the MILPOptimiser class and feed in a Pandas DataFrame containing player gameweek points projection data for the required time interval. One can also construct the required input data using the FplAPIData and FplXPtsForecastData classes. 

These classes are made available at the top-level:

```python
from fpl_optimiser import FplAPIData, FplXPtsForecastData, MILPOptimiser
```

Player gameweek projection data can be created using the FplAPIData and FplXPtsForecastData classes:

```python
import pandas as pd

GAMEWEEK = 7  # Define starting gameweek for projection.
FIRST_RUN = False # Data is extracted directly from the FPL API for the first run otherwise the extracted data is read from within the repository. 

# 1). Fetch required data from official FPL API.
api_data = FplAPIData()
if FIRST_RUN:
  # Extract relevant data from the FPL API and save to `data/official_api_data/`
    api_data.get_gw_player_data(gameweek=GAMEWEEK)  # Static player data for the specified gameweek
    api_data.get_gw_team_lineup_data(gameweek=GAMEWEEK-1)   # User's team line-up data in the week prior to the specified gameweek 

player_data_df = api_data.read_gw_player_data(gameweek=GAMEWEEK)  # Read CSV file extracted from API
player_data_df = player_data_df[~player_data_df["position"].isna()]  # Remove managers from the dataset

# 2). Format and extract xPts data.
xpts_data = FplXPtsForecastData()
xpts_data.get_gw_player_forecast_data(gameweek=GAMEWEEK)
xpts_df = xpts_data.gw_forecast_df

# Merge API and xPts data together.
gw_df = pd.merge(player_data_df, xpts_df.drop(columns=["position", "cost"]), how="left", on="name").reset_index(drop=True)
```

The optimiser can be defined by supplying player gameweek points projection data and any additional arguments to the MILPOptimiser constructor: 
```python
gw_optimiser = MILPOptimiser(gw_df,  # Player gameweek projection Pandas DataFrame.
                             start_gameweek = GAMEWEEK,   # Starting point for gameweek projection.
                             gameweeks = 3,  # Must be aligned to the number of gameweek projection 
                             use_existing_team = EXISTING_TEAM,  # Specify whether optimisation is performed assuming an existing team or not.
                             )

# Perform optimisation.
gw_optimiser.calulate_optimal_team()
```

### 4. Output

The optimiser will output the optimal team selection by default and present each player selection and their expected points by gameweek. Decisions regarding formation, transfers, captaincy and starting 11 vs. bench selection are also displayed. This output can be accessed using the results_df attribute of the optimiser object, this can then be saved as a CSV file for further analysis.

Here is example report that is displayed to the console:
```Calculating a 3-gameweek forecast, starting from GW: 23...
Gameweek 23:
Optimal team:
      id        name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
30   443        Sels      GKP  NOT          0.0      22     0.578947                 17.8     94          5.0        23   3.4      Outfield    False         False
9    399        Hall      DEF  NEW          0.0      19     0.500000                 26.7     83          5.1        23   3.8      Outfield    False         False
44   339      Virgil      DEF  LIV          0.0      21     0.552632                 22.7     91          6.4        23   4.9      Outfield    False         False
49   255    Robinson      DEF  FUL          0.0      22     0.578947                 26.6     89          5.1        23   3.3      Outfield    False         False
3    328     M.Salah      MID  LIV          0.0      21     0.552632                 70.7     89         13.7        23   9.6      Outfield     True         False
4    398      Gordon      MID  NEW          0.0      20     0.526316                 26.1     85          7.7        23   5.2      Outfield    False         False
7     71    Kluivert      MID  BOU          0.0      17     0.447368                  8.0     84          5.6        23   5.3      Outfield    False         False
10   182      Palmer      MID  CHE          0.0      22     0.578947                 66.1     88         11.3        23   4.7      Outfield    False         False
1    401        Isak      FWD  NEW          0.0      20     0.526316                 59.8     88          9.5        23   6.5      Outfield    False          True
29   252        Raúl      FWD  FUL          0.0      17     0.447368                 11.7     74          5.7        23   4.5      Outfield    False         False
41   129  João Pedro      FWD  BRI          0.0      14     0.368421                 15.0     79          5.6        23   4.7      Outfield    False         False
263  521   Fabianski      GKP  WHM          0.0      12     0.315789                 16.8     87          4.1        23   3.0         Bench    False         False
32   422        Aina      DEF  NOT          0.0      22     0.578947                 29.3     89          5.4        23   2.5         Bench    False         False
50   573  Milenković      DEF  NOT          0.0      21     0.552632                  8.7     88          4.8        23   2.2         Bench    False         False
8    364        Amad      MID  MUN          0.0      15     0.394737                 23.5     85          5.6        23   3.8         Bench    False         False

Formation: 3,4,3
Total team cost: 100.6
   (o/w Outfield): 80.7
   (o/w Bench): 19.9
Total expected points (excl. Captain): 55.900000000000006
Total expected points (incl. Captain): 65.5
Captain: M.Salah
Vice-Captain: Isak
Transfered out: B.Fernandes
Transferred in: Kluivert
Players benched: Aina, Amad, Milenković
Players promoted: João Pedro, Robinson, Virgil

Gameweek 24:
Optimal team:
      id              name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
30   443              Sels      GKP  NOT          0.0      22     0.578947                 17.8     94          4.9        24   3.4      Outfield    False         False
5    311  Alexander-Arnold      DEF  LIV          0.0      19     0.500000                 30.6     77          7.1        24   8.3      Outfield    False          True
9    399              Hall      DEF  NEW          0.0      19     0.500000                 26.7     83          5.0        24   3.4      Outfield    False         False
44   339            Virgil      DEF  LIV          0.0      21     0.552632                 22.7     91          6.2        24   7.1      Outfield    False         False
3    328           M.Salah      MID  LIV          0.0      21     0.552632                 70.7     89         13.4        24  12.8      Outfield     True         False
4    398            Gordon      MID  NEW          0.0      20     0.526316                 26.1     85          7.5        24   4.8      Outfield    False         False
7     71          Kluivert      MID  BOU          0.0      17     0.447368                  8.0     84          5.4        24   3.7      Outfield    False         False
8    364              Amad      MID  MUN          0.0      15     0.394737                 23.5     85          5.5        24   4.4      Outfield    False         False
10   182            Palmer      MID  CHE          0.0      22     0.578947                 66.1     88         11.1        24   6.9      Outfield    False         False
1    401              Isak      FWD  NEW          0.0      20     0.526316                 59.8     88          9.3        24   5.9      Outfield    False         False
41   129        João Pedro      FWD  BRI          0.0      14     0.368421                 15.0     79          5.4        24   3.7      Outfield    False         False
263  521         Fabianski      GKP  WHM          0.0      12     0.315789                 16.8     87          4.0        24   2.9         Bench    False         False
32   422              Aina      DEF  NOT          0.0      22     0.578947                 29.3     89          5.3        24   3.2         Bench    False         False
49   255          Robinson      DEF  FUL          0.0      22     0.578947                 26.6     89          5.0        24   2.2         Bench    False         False
29   252              Raúl      FWD  FUL          0.0      17     0.447368                 11.7     74          5.5        24   3.5         Bench    False         False

Formation: 3,5,2
Total team cost: 100.6
   (o/w Outfield): 80.8
   (o/w Bench): 19.8
Total expected points (excl. Captain): 64.4
Total expected points (incl. Captain): 77.20000000000002
Captain: M.Salah
Vice-Captain: Alexander-Arnold
Transfered out: Milenković
Transferred in: Alexander-Arnold
Players benched: Raúl, Robinson
Players promoted: Amad

Gameweek 25:
Optimal team:
      id              name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
30   443              Sels      GKP  NOT          0.0      22     0.578947                 17.8     94          4.8        25   3.3      Outfield    False         False
5    311  Alexander-Arnold      DEF  LIV          0.0      19     0.500000                 30.6     77          6.9        25   5.1      Outfield    False         False
44   339            Virgil      DEF  LIV          0.0      21     0.552632                 22.7     91          6.0        25   4.2      Outfield    False         False
49   255          Robinson      DEF  FUL          0.0      22     0.578947                 26.6     89          4.9        25   3.7      Outfield    False         False
3    328           M.Salah      MID  LIV          0.0      21     0.552632                 70.7     89         13.1        25   7.9      Outfield     True         False
7     71          Kluivert      MID  BOU          0.0      17     0.447368                  8.0     84          5.2        25   4.8      Outfield    False         False
10   182            Palmer      MID  CHE          0.0      22     0.578947                 66.1     88         10.9        25   5.0      Outfield    False         False
110  503               Son      MID  TOT          0.0      17     0.447368                  5.1     83          9.4        25   6.4      Outfield    False          True
1    401              Isak      FWD  NEW          0.0      20     0.526316                 59.8     88          9.1        25   4.0      Outfield    False         False
29   252              Raúl      FWD  FUL          0.0      17     0.447368                 11.7     74          5.3        25   5.0      Outfield    False         False
41   129        João Pedro      FWD  BRI          0.0      14     0.368421                 15.0     79          5.2        25   3.5      Outfield    False         False
263  521         Fabianski      GKP  WHM          0.0      12     0.315789                 16.8     87          3.9        25   3.2         Bench    False         False
9    399              Hall      DEF  NEW          0.0      19     0.500000                 26.7     83          4.9        25   1.9         Bench    False         False
32   422              Aina      DEF  NOT          0.0      22     0.578947                 29.3     89          5.2        25   2.6         Bench    False         False
8    364              Amad      MID  MUN          0.0      15     0.394737                 23.5     85          5.4        25   3.5         Bench    False         False

Formation: 3,4,3
Total team cost: 100.20000000000002
   (o/w Outfield): 80.8
   (o/w Bench): 19.4
Total expected points (excl. Captain): 52.9
Total expected points (incl. Captain): 60.8
Captain: M.Salah
Vice-Captain: Son
Transfered out: Gordon
Transferred in: Son
Players benched: Hall, Amad
Players promoted: Raúl, Robinson

Optimisation process complete!
Time taken: 9.59 seconds
```
## Contributing

If you encounter bugs, want to suggest improvements, or have new features in mind, feel free to fork the repository and submit a pull request.

- Fork the repository and clone it locally.
- Create a new branch for your feature or bug fix.
- Write tests to ensure the new feature works as expected.
- Submit a pull request with a clear description of your changes.
