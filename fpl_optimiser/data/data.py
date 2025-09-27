import os
import requests
import pandas as pd
import numpy as np

from ..utils import YAMLFile, DATA_DIR
from .enrichment import *

class FplAPIData:
    """
    Class representing data from the official FPL API.
    """

    def __init__(self, config: YAMLFile = YAMLFile()) -> None:
        self.config = config
        season_years  = self.config.season.split("/")
        self.season_filename_prefix = "_".join([year[-2:] for year in season_years])
        self.player_gw_data_df = None
        self.team_used_gw_df = None
        self.directory = os.path.join(DATA_DIR, "official_api_data", "_".join(season_years))
        self.auth_cookie = {"Cookie": self.config.fpl_api_cookie_auth}

    def get_gw_team_lineup_data(self, gameweek: int, save_to_disk: bool = True) -> pd.DataFrame:
        """
        Returns a pd.DataFrame object containing the team used in the specified gameweek. 
        """

        api_base_url, team_id = self.config.fpl_api_base_url, self.config.FPL_TEAM_ID
        api_endpoint = f"{api_base_url}/entry/{team_id}/event/{gameweek}/picks"
        try:
            response = requests.get(api_endpoint, cookies=self.auth_cookie)
            response.raise_for_status()
            gw_team_data = response.json()
        except requests.exceptions.RequestException as req_err:
            print(f"Error: {req_err}")

        df = pd.DataFrame(data=gw_team_data["picks"])
        
        # Extract and merge player values to dataframe containing team data.
        all_player_data = self.get_gw_player_data(gameweek, save_to_disk=False)
        player_data = all_player_data[all_player_data["id"].isin(set(df["element"]))].copy()
        player_data.rename(columns={"id": "element", "now_cost": "value"}, inplace=True)

        df = pd.merge(df, player_data[["element", "name", "value"]], on="element", how="left")
        self.team_used_gw_df = df.copy()

        if save_to_disk:
            # Prevent user from saving if file already exists.
            # We should raise a warning and ask for confirmation.
            filename = f"FPL {self.season_filename_prefix} season - team GW{gameweek}.csv"
            df.to_csv(os.path.join(self.directory, filename), index=False)
        
        return df
    
    def read_gw_team_lineup_data(self, gameweek: int) -> pd.DataFrame:
        """
        Reads the CSV file containing the team selection for the given gameweek.
        The CSV file is read from the data/official_api_data/ directory.
        Returns a pd.DataFrame object containing the dataset.
        """
        
        filename = f"FPL {self.season_filename_prefix} season - team GW{gameweek}.csv"
        filepath = os.path.join(self.directory, filename)

        # Check if file exists.
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: {filepath} does not exist!")

        # Try to open file, raise ValueError if unable to.
        try:
            df = pd.read_csv(filepath)
        except ValueError:
            print(f"Error: {filepath} could not be opened")

        return df
    
    def get_gw_player_data(self, gameweek: int, save_to_disk: bool = True) -> pd.DataFrame:
        """ 
        Returns a pd.DataFrame object containing player data for the specified gameweek.
        Data is cleaned and pre-processed, ready for optimisation.
        Default keyword arg "save_to_disk" - saves to the project folder by default.
        """
        
        api_endpoint = f"{self.config.fpl_api_base_url}/bootstrap-static/"

        # Retrieve data from API using requests and JSON
        try:
            response = requests.get(api_endpoint, cookies=self.auth_cookie)
            response.raise_for_status()
            player_api_data = response.json()
        except requests.exceptions.RequestException as req_err:
            print(f"Error: {req_err}")

        player_data = dict()
        for field in player_api_data["elements"][0].keys():
            all_field_values = []
            for curr_data in player_api_data["elements"]:
                all_field_values.append(curr_data[field])
            
            player_data[field] = all_field_values

        df = pd.DataFrame(data=player_data, columns=player_data.keys())
        
        key_cols = [
            "id", "web_name", "position", "team", "prob_injury", "starts", "starts_perc",
            "starts_per_90", "minutes", "selected_by_percent", "ep_next", "now_cost", "gw"
        ]
        # Calculate further measures and restrict attention to key columns.
        df["gw"] = gameweek
        df["ep_next"] = df["ep_next"].astype("float64").fillna(0)
        df["chance_of_playing_next_round"] = df["chance_of_playing_next_round"].fillna(100)
        df["prob_injury"] = 1 - (df["chance_of_playing_next_round"]/ 100)
        df["starts_perc"] = df["starts"] / 38
        df["now_cost"] = df["now_cost"] / 10

        position_mapping = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        df["position"] = df["element_type"].map(position_mapping)

        team_mapping = {
            "24_25" : {1: "ARS", 2: "AST", 3: "BOU", 4: "BRE", 5:"BRI",
                       6: "CHE", 7: "CRY", 8: "EVE", 9: "FUL", 10: "IPS", 
                       11: "LEI", 12: "LIV", 13: "MCI", 14: "MUN", 15: "NEW",
                       16: "NOT", 17: "SOU", 18: "TOT", 19: "WHM", 20: "WOL"},
            "25_26" : {1: "ARS", 2: "AST", 3: "BUR", 4: "BOU", 5:"BRE",
                       6: "BRI", 7: "CHE", 8: "CRY", 9: "EVE", 10: "FUL", 
                       11: "LEE", 12: "LIV", 13: "MCI", 14: "MUN", 15: "NEW",
                       16: "NOT", 17: "SUN", 18: "TOT", 19: "WHM", 20: "WOL"}
        }
        if self.season_filename_prefix in team_mapping:
            curr_team_mapping = team_mapping[self.season_filename_prefix]
            df["team"] = df["team"].map(curr_team_mapping)

        # Keep relevant columns and sort dataframe.
        sum_df = df[key_cols].copy()
        sum_df.sort_values(by=["ep_next", "now_cost"], ascending=False).reset_index(drop=True)

        # Apply data enrichment to remove duplicate names.
        if self.season_filename_prefix in fpl_api_id_name_map:
            name_mapping = fpl_api_id_name_map[self.season_filename_prefix]
            sum_df["web_name"] = sum_df["id"].map(name_mapping).fillna(sum_df["web_name"])
        
        sum_df.rename(columns={"web_name": "name"}, inplace=True)
        self.player_gw_data_df = sum_df.copy()

        if save_to_disk:
            filename = f"FPL {self.season_filename_prefix} season - official API GW{gameweek} data.csv"
            sum_df.to_csv(os.path.join(self.directory, filename), index=False)

        return sum_df

    def read_gw_player_data(self, gameweek: int) -> pd.DataFrame:
        """
        Reads the official FPL API dataset, from disk, for the specified gameweek.
        Returns a Pandas DataFrame object containing the dataset.
        """
        
        filename = f"FPL {self.season_filename_prefix} season - official API GW{gameweek} data.csv"
        filepath = os.path.join(self.directory, filename)

        # Check if file exists.
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: {filepath} does not exist!")

        # Try to open file, raise ValueError if unable to.
        try:
            df = pd.read_csv(filepath)
        except ValueError:
            print(f"Error: {filepath} could not be opened")

        return df
    
    def get_team_value(self, gameweek: int) -> tuple[float, float]:
        """
        Returns a tuple containing the total squad value and total money in the bank 
        for the specified gameweek.
        """

        api_base_url, team_id = self.config.fpl_api_base_url, self.config.FPL_TEAM_ID
        api_endpoint = f"{api_base_url}/entry/{team_id}/event/{gameweek}/picks/"
        try:
            response = requests.get(api_endpoint, cookies=self.auth_cookie)
            response.raise_for_status()
            gw_team_data = response.json()
        except requests.exceptions.RequestException as req_err:
            print(f"Error: {req_err}")

        # Extract budget values and convert to £mn (Millions).
        team_value = gw_team_data["entry_history"]["value"] / 10.0 
        bank = gw_team_data["entry_history"]["bank"] / 10.0

        return (team_value-bank, bank)

class FplXPtsForecastData:
    """
    Class representing FPL xPts forecast data.
    """

    def __init__(self, config: YAMLFile = YAMLFile()) -> None:
        self.config = config
        season_years  = self.config.season.split("/")
        self.season_filename_prefix = "_".join([year[-2:] for year in season_years])
        self.gw_forecast_df = None
        self.directory = os.path.join(DATA_DIR, "fpl_xpts_forecast_data", "_".join(season_years))
    
    def get_gw_player_forecast_data(self, gameweek: int, save_to_disk: bool = True) -> pd.DataFrame:
        """
        Loads and transforms a CSV file within self.directory/raw, containing the raw (unformatted)
        expected points and minutes played forecast player data for a specified number of gameweeks. 
        Return a Pandas DataFrame object containing the formatted data. 
        Will save a copy of the file to self.directory/clean by default.
        """
        raw_filename = f"FPL xPts forecast GW{gameweek} raw data.csv"
        filepath = os.path.join(self.directory, "raw", raw_filename)
        raw_data = pd.read_csv(filepath, encoding="utf-8")
        rows = raw_data.shape[0]
        
        extract_idx = lambda start_idx: [i for i in range(start_idx, rows + 1, 4) if i <= rows]
        player_names = raw_data.loc[extract_idx(0), "col1"].dropna().reset_index(drop=True)
        pos_and_costs = raw_data.loc[extract_idx(1), :]["col1"].dropna().reset_index(drop=True)
        proj_exp_points = raw_data.iloc[extract_idx(2), 1:5].reset_index(drop=True)
        proj_exp_mins = raw_data.iloc[extract_idx(2), 5:].reset_index(drop=True)
        
        df = pd.concat([player_names, pos_and_costs, proj_exp_mins, proj_exp_points], ignore_index=True, axis=1)
        num_of_gameweeks = 3 if self.season_filename_prefix == "24_25" else 4
        xmins_cols = [f"xmins_gw{gameweek + i}" for i in range(0, num_of_gameweeks)]
        xpts_cols = [f"xpts_gw{gameweek + i}" for i in range(0, num_of_gameweeks)]
        df.columns = ["name", "position_and_costs"] + xmins_cols + xpts_cols
        df[["position", "cost"]] = df["position_and_costs"].str.split(" ", expand=True)
        df = df[["name", "position", "cost"] + xmins_cols + xpts_cols]
        df["position"] = df["position"].map({"GK": "GKP", "DF": "DEF", "MD":"MID", "FW": "FWD"})
        df["cost"] = df["cost"].astype("float32")
        df["cost"] = np.where(df["cost"] == 99.9, 0, df["cost"])
        df[xmins_cols] = df[xmins_cols].astype("int32").fillna(0)
        df[xpts_cols] = df[xpts_cols].fillna(0)
        
        # Data enrichment
        if self.season_filename_prefix in fpl_xPts_forecast_name_map:
            name_mapping = fpl_xPts_forecast_name_map[self.season_filename_prefix]
            df["name"] = df["name"].replace(name_mapping)
        
        if self.season_filename_prefix in fpl_xPts_forecast_name_pos_map:
            name_pos_mapping = fpl_xPts_forecast_name_pos_map[self.season_filename_prefix]
            name_pos_mapper = lambda x: name_pos_mapping.get((x["name"], x["position"]), x["name"])
            df["name"] = df.apply(name_pos_mapper, axis=1)
       
        if save_to_disk:
            filename = f"FPL {self.season_filename_prefix} season - xPts forecast GW{gameweek} data.csv"
            df.to_csv(os.path.join(self.directory, "clean", filename) ,index=False)
        
        self.gw_forecast_df = df.copy()
        return df