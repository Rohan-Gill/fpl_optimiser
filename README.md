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

**FPL Optimiser** is a Python tool designed to assist managers in making data-driven decisions regarding optimal team selection in the [Fantasy Premier League (FPL)](https://fantasy.premierleague.com/) football game. The optimiser formulates and solves an [**integer linear programming** (ILP)](https://en.wikipedia.org/wiki/Integer_programming) problem, where forecasted points over a specified gameweek interval are maximised subject to a set of constraints based on FPL game rules. These constraints include player positions, budget limits, and team selection rules, all of which are critical to ensuring the team's validity in the game. 

This tool enables the optimisation of team selection for upcoming gameweeks using forecast data and also provides the ability to calculate optimal team selections for historic gameweeks using actual data observed in previous gameweeks. The combination of forecast-based and actual data-driven optimisation makes FPL Optimiser a versatile tool for FPL managers aiming for the best possible team composition.

## Key Features

- **Multi-Gameweek Optimisation**: Supports optimisation over multiple gameweeks to account for longer-term planning and strategy.
- **Real-Time Calculation**: Performs efficient calculations to deliver optimal team selection in real time, providing users with actionable insights immediately.
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
season: "2025/2026"  # Update the season

fpl_api_base_url: "https://fantasy.premierleague.com/api"  # Base URL for FPL API

FPL_TEAM_ID: 1234567  # Update to your FPL team ID

# Authentication cookie for accessing the FPL API.
fpl_api_cookie_auth: >+
  "Your authentication cookie"  # Update with your cookie
```

### 2. Data preparation

The optimiser requires player gameweek points projection data. This can be input to the solver as a Pandas DataFrame object or can be sourced from the following sub-directories:

- **`data/fpl_xpts_forecast_data/`**
- **`data/official_api_data/`**
- **`data/actuals/`**

Ensure that the appropriate data files are present in these directories before running the optimiser. If you are using your own data, ensure the format and structure of the data is aligned to that obtained in section 3 (see below).

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
EXISTING_TEAM = True  # Specify whether optimisation is performed assuming an existing team or not.

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
                             use_existing_team = EXISTING_TEAM,
                             )

# Perform optimisation.
gw_optimiser.calulate_optimal_team()
```

### 4. Output

The optimiser will output the optimal team selection by default and present each player selection and their expected points by gameweek. Decisions regarding formation, transfers, captaincy and starting 11 vs. bench selection are also displayed. This output can be accessed using the results_df attribute of the optimiser object, this can then be saved as a CSV file for further analysis.

Here is example report that is displayed to the console:
```
Calculating a 4-gameweek forecast, starting from GW: 6...
Gameweek 6:
Optimal team:
      id         name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
119   67     Petrović      GKP  BOU         0.00       5     0.131579                  5.5     93          4.5         6   3.7      Outfield    False         False
124   72       Senesi      DEF  BOU         0.00       5     0.131579                 17.9     90          4.8         6   4.2      Outfield    False         False
414  373       Virgil      DEF  LIV         0.00       5     0.131579                 29.8     89          6.1         6   4.2      Outfield    False         False
559  508   N.Williams      DEF  NOT         0.00       5     0.131579                  6.2     88          4.9         6   4.7      Outfield    False         False
134   82      Semenyo      MID  BOU         0.00       5     0.131579                 50.9     88          7.7         6   4.8      Outfield    False         False
322  299       Ndiaye      MID  EVE         0.00       5     0.131579                 11.2     73          6.5         6   4.9      Outfield    False         False
458  414        Foden      MID  MCI         0.00       2     0.052632                  5.8     73          8.1         6   5.5      Outfield    False          True
469  427    Reijnders      MID  MCI         0.00       5     0.131579                 34.3     82          5.7         6   4.8      Outfield    False         False
495  449  B.Fernandes      MID  MUN         0.00       5     0.131579                 21.1     87          9.0         6   5.1      Outfield    False         False
31   666     Gyökeres      FWD  ARS         0.00       5     0.131579                 25.4     81          9.0         6   4.7      Outfield    False         False
472  430      Haaland      FWD  MCI         0.00       5     0.131579                 48.9     84         14.3         6   7.3      Outfield     True         False
114  470     Dúbravka      GKP  BUR         0.00       5     0.131579                 34.4     93          4.0         6   2.8         Bench    False         False
39    36         Cash      DEF  AST         0.00       5     0.131579                  5.3     79          4.6         6   3.7         Bench    False         False
487  441        Dorgu      DEF  MUN         0.00       4     0.105263                  5.2     85          4.5         6   3.4         Bench    False         False
269  252    Marc Guiu      FWD  CHE         0.25       0     0.000000                  8.6      6          4.3         6   0.5         Bench    False         False

Formation: 3,5,2
Total budget: 100.0
 o/w Funds in bank: 2.0
 o/w Team cost: 98.0
   o/w Outfield: 80.6
   o/w Bench: 17.4
Total expected points (excl. Captain): 53.9
Total expected points (incl. Captain): 61.2
Captain: Haaland
Vice-Captain: Foden
Transfered out: N/A
Transferred in: N/A
Players benched: N/A
Players promoted: N/A

Gameweek 7:
Optimal team:
      id         name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
119   67     Petrović      GKP  BOU         0.00       5     0.131579                  5.5     91          4.4         7   3.7      Outfield    False         False
39    36         Cash      DEF  AST         0.00       5     0.131579                  5.3     81          4.5         7   4.2      Outfield    False         False
124   72       Senesi      DEF  BOU         0.00       5     0.131579                 17.9     88          4.7         7   4.3      Outfield    False         False
414  373       Virgil      DEF  LIV         0.00       5     0.131579                 29.8     88          6.0         7   3.8      Outfield    False         False
487  441        Dorgu      DEF  MUN         0.00       4     0.105263                  5.2     78          4.5         7   4.6      Outfield    False         False
14    16         Saka      MID  ARS         0.00       2     0.052632                  4.9     81          9.8         7   5.9      Outfield    False         False
134   82      Semenyo      MID  BOU         0.00       5     0.131579                 50.9     83          7.6         7   4.9      Outfield    False         False
322  299       Ndiaye      MID  EVE         0.00       5     0.131579                 11.2     76          6.4         7   4.4      Outfield    False         False
495  449  B.Fernandes      MID  MUN         0.00       5     0.131579                 21.1     88          8.9         7   6.5      Outfield     True         False
31   666     Gyökeres      FWD  ARS         0.00       5     0.131579                 25.4     81          8.9         7   6.5      Outfield    False          True
472  430      Haaland      FWD  MCI         0.00       5     0.131579                 48.9     86         14.1         7   5.9      Outfield    False         False
114  470     Dúbravka      GKP  BUR         0.00       5     0.131579                 34.4     91          3.9         7   3.0         Bench    False         False
559  508   N.Williams      DEF  NOT         0.00       5     0.131579                  6.2     85          4.8         7   2.9         Bench    False         False
469  427    Reijnders      MID  MCI         0.00       5     0.131579                 34.3     81          5.6         7   3.9         Bench    False         False
269  252    Marc Guiu      FWD  CHE         0.25       0     0.000000                  8.6     13          4.3         7   0.9         Bench    False         False

Formation: 4,4,2
Total budget: 98.6
 o/w Funds in bank: 0.2
 o/w Team cost: 98.4
   o/w Outfield: 79.8
   o/w Bench: 18.6
Total expected points (excl. Captain): 54.7
Total expected points (incl. Captain): 61.2
Captain: B.Fernandes
Vice-Captain: Gyökeres
Transfered out: Foden
Transferred in: Saka
Players benched: N.Williams, Reijnders
Players promoted: Dorgu, Cash

Gameweek 8:
Optimal team:
      id         name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
114  470     Dúbravka      GKP  BUR         0.00       5     0.131579                 34.4     92          3.9         8   3.6      Outfield    False         False
85   191       Estève      DEF  BUR         0.00       5     0.131579                 15.0     89          4.0         8   3.8      Outfield    False         False
124   72       Senesi      DEF  BOU         0.00       5     0.131579                 17.9     84          4.6         8   3.5      Outfield    False         False
414  373       Virgil      DEF  LIV         0.00       5     0.131579                 29.8     83          5.9         8   4.1      Outfield    False         False
14    16         Saka      MID  ARS         0.00       2     0.052632                  4.9     83          9.7         8   4.7      Outfield    False          True
134   82      Semenyo      MID  BOU         0.00       5     0.131579                 50.9     80          7.5         8   3.9      Outfield    False         False
322  299       Ndiaye      MID  EVE         0.00       5     0.131579                 11.2     80          6.3         8   3.6      Outfield    False         False
469  427    Reijnders      MID  MCI         0.00       5     0.131579                 34.3     75          5.5         8   3.9      Outfield    False         False
495  449  B.Fernandes      MID  MUN         0.00       5     0.131579                 21.1     86          8.8         8   4.1      Outfield    False         False
31   666     Gyökeres      FWD  ARS         0.00       5     0.131579                 25.4     71          8.8         8   4.4      Outfield    False         False
472  430      Haaland      FWD  MCI         0.00       5     0.131579                 48.9     80         13.9         8   5.9      Outfield     True         False
119   67     Petrović      GKP  BOU         0.00       5     0.131579                  5.5     92          4.3         8   3.3         Bench    False         False
39    36         Cash      DEF  AST         0.00       5     0.131579                  5.3     80          4.4         8   2.8         Bench    False         False
487  441        Dorgu      DEF  MUN         0.00       4     0.105263                  5.2     68          4.4         8   1.9         Bench    False         False
269  252    Marc Guiu      FWD  CHE         0.25       0     0.000000                  8.6      6          4.3         8   0.5         Bench    False         False

Formation: 3,5,2
Total budget: 97.2
 o/w Funds in bank: 0.9
 o/w Team cost: 96.3
   o/w Outfield: 78.9
   o/w Bench: 17.4
Total expected points (excl. Captain): 45.5
Total expected points (incl. Captain): 51.4
Captain: Haaland
Vice-Captain: Saka
Transfered out: N.Williams
Transferred in: Estève
Players benched: Dorgu, Cash, Petrović
Players promoted: Dúbravka, Reijnders

Gameweek 9:
Optimal team:
      id         name position team  prob_injury  starts  starts_perc  selected_by_percent  xMins  player_cost  gameweek  xPts position_type  captain  vice_captain
119   67     Petrović      GKP  BOU         0.00       5     0.131579                  5.5     92          4.2         9   3.7      Outfield    False         False
85   191       Estève      DEF  BUR         0.00       5     0.131579                 15.0     85          4.0         9   3.5      Outfield    False         False
124   72       Senesi      DEF  BOU         0.00       5     0.131579                 17.9     81          4.5         9   4.0      Outfield    False         False
414  373       Virgil      DEF  LIV         0.00       5     0.131579                 29.8     82          5.8         9   4.0      Outfield    False         False
134   82      Semenyo      MID  BOU         0.00       5     0.131579                 50.9     76          7.4         9   4.7      Outfield    False         False
253  235       Palmer      MID  CHE         1.00       2     0.052632                 15.2     70         10.5         9   6.0      Outfield     True         False
322  299       Ndiaye      MID  EVE         0.00       5     0.131579                 11.2     69          6.2         9   4.0      Outfield    False         False
469  427    Reijnders      MID  MCI         0.00       5     0.131579                 34.3     75          5.4         9   3.4      Outfield    False         False
495  449  B.Fernandes      MID  MUN         0.00       5     0.131579                 21.1     86          8.7         9   5.3      Outfield    False          True
31   666     Gyökeres      FWD  ARS         0.00       5     0.131579                 25.4     64          8.7         9   4.5      Outfield    False         False
472  430      Haaland      FWD  MCI         0.00       5     0.131579                 48.9     77         13.7         9   4.9      Outfield    False         False
114  470     Dúbravka      GKP  BUR         0.00       5     0.131579                 34.4     91          3.8         9   3.3         Bench    False         False
39    36         Cash      DEF  AST         0.00       5     0.131579                  5.3     75          4.3         9   2.6         Bench    False         False
487  441        Dorgu      DEF  MUN         0.00       4     0.105263                  5.2     67          4.3         9   2.9         Bench    False         False
269  252    Marc Guiu      FWD  CHE         0.25       0     0.000000                  8.6     10          4.3         9   0.9         Bench    False         False

Formation: 3,5,2
Total budget: 95.8
 o/w Funds in bank: 0.0
 o/w Team cost: 95.8
   o/w Outfield: 79.1
   o/w Bench: 16.7
Total expected points (excl. Captain): 48.0
Total expected points (incl. Captain): 54.0
Captain: Palmer
Vice-Captain: B.Fernandes
Transfered out: Saka
Transferred in: Palmer
Players benched: Dúbravka
Players promoted: Petrović

Optimisation process complete!
Time taken: 83.07 seconds
```
## Contributing

If you encounter bugs, want to suggest improvements, or have new features in mind, feel free to fork the repository and submit a pull request.

- Fork the repository and clone it locally.
- Create a new branch for your feature or bug fix.
- Write tests to ensure the new feature works as expected.
- Submit a pull request with a clear description of your changes.
