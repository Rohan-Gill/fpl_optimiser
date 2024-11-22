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
        self.season_label = "_".join([year[-2:] for year in self.config.season.split("/")])
        self.player_gw_data_df = None
        self.team_used_gw_df = None
        self.directory = os.path.join(DATA_DIR, "official_api_data")
        self.auth_cookie = {"Cookie": self.config.fpl_api_cookie_auth}
    def get_gw_team_lineup_data(self, gameweek: int, save_to_disk: bool = True) -> pd.DataFrame:
        """
        Generates a pd.DataFrame object that represents the team used in the specified (historic) gameweek. 
        """

        api_endpoint = f"{self.config.fpl_api_base_url}/entry/{self.config.FPL_TEAM_ID}/event/{gameweek}/picks"
        try:
            response = requests.get(api_endpoint, cookies=self.auth_cookie)
            response.raise_for_status()
            gw_team_data = response.json()
        except requests.exceptions.RequestException as req_err:
            print(f"Error: {req_err}")

        df = pd.DataFrame(data=gw_team_data["picks"])
        
        if save_to_disk:
            filename = f"FPL {self.season_label} season - team GW{gameweek}.csv"
            df.to_csv(os.path.join(self.directory, filename) ,index=False)
        
        self.team_used_gw_df = df.copy()
        return df
    
    def read_gw_team_lineup_data(self, gameweek: int) -> pd.DataFrame:
        """
        Reads the .csv file containing the team selection for the given (historic) gameweek.
        Returns a Pandas DataFrame object containing the dataset.
        """
        
        filename = f"FPL {self.season_label} season - team GW{gameweek}.csv"
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
        Generates a pd.DataFrame object containing data from the official FPL API for the specified gameweek.
        Data is cleaned and pre-processed, ready for optimisation.
        The function provides a default keyword arg "save_to_disk", which saves the current view to the project folder by default.
        Default is True since API data will change wrt. time so a method call at a future data for a given gameweek will produce different data.
        """
        
        api_endpoint = f"{self.config.fpl_api_base_url}/bootstrap-static/"

        # Retrieve (dict) data from API using requests and JSON
        try:
            response = requests.get(api_endpoint, cookies=self.auth_cookie)
            response.raise_for_status()
            player_api_data = response.json()
        except requests.exceptions.RequestException as req_err:
            print(f"Error: {req_err}")

        player_data = dict()
        for field in player_api_data["elements"][0].keys():
            # Player_data is a dict containing information for players present within the specified gameweek
            all_field_values = []
            for curr_data in player_api_data["elements"]:
                all_field_values.append(curr_data[field])
            
            player_data[field] = all_field_values

        player_data_df = pd.DataFrame(data=player_data, columns=player_data.keys())
        player_data_df["gw"] = gameweek
        
        # Calculate further measures and restrict attention to key columns.
        key_cols = ["id", "web_name", "position", "team", "prob_injury", "starts", "starts_perc", "starts_per_90", "minutes", "selected_by_percent", "ep_next", "now_cost", "gw"]
        player_data_df["ep_next"] = player_data_df["ep_next"].astype("float64").fillna(0)
        player_data_df["prob_injury"] = 1 - (player_data_df["chance_of_playing_next_round"].fillna(100) / 100)
        player_data_df["starts_perc"] = player_data_df["starts"] / 38
        player_data_df["now_cost"] = player_data_df["now_cost"] / 10
        player_data_df["position"] = player_data_df["element_type"].map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})
        team_mapping = {1: "ARS", 2: "AST", 3: "BOU", 4: "BRE", 5:"BRI", 6: "CHE", 7: "CRY", 8: "EVE", 9: "FUL", 10: "IPS",
                        11: "LEI", 12: "LIV", 13: "MCI", 14: "MUN", 15: "NEW", 16: "NOT", 17: "SOU", 18: "TOT", 19: "WHM", 20: "WOL"}
        player_data_df["team"] = player_data_df["team"].map(team_mapping)
        summary_df = player_data_df[key_cols].sort_values(by=["ep_next", "now_cost"], ascending=False).reset_index(drop=True)

        # Apply data enrichment to remove duplicate names.
        summary_df["web_name"] = summary_df["id"].map(fpl_api_id_name_map).fillna(summary_df["web_name"])
        summary_df.rename(columns={"web_name": "name"}, inplace=True)

        if save_to_disk:
            filename = f"FPL {self.season_label} season - official API GW{gameweek} data.csv"
            summary_df.to_csv(os.path.join(DATA_DIR, "official_api_data", filename), index=False)

        self.player_gw_data_df = summary_df.copy()
        return summary_df

    def read_gw_player_data(self, gameweek: int) -> pd.DataFrame:
        """
        Reads the official FPL API dataset, from disk, for the specified gameweek.
        Returns a Pandas DataFrame object containing the dataset.
        """
        
        filename = filename = f"FPL {self.season_label} season - official API GW{gameweek} data.csv"
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


class FplXPtsForecastData:
    """
    Class representing FPL xPts forecast data.
    """

    def __init__(self, config: YAMLFile = YAMLFile()) -> None:
        self.config = config
        self.season_label = "_".join([year[-2:] for year in self.config.season.split("/")])
        self.gw_forecast_df = None
        self.directory = os.path.join(DATA_DIR, "fpl_xpts_forecast_data")
    
    def get_gw_player_forecast_data(self, gameweek: int, save_to_disk: bool = True) -> pd.DataFrame:
        """
        Loads and transforms the raw data .csv file within self.directory/raw. 
        Return a Pandas DataFrame object containing the transformed data and 
        will also save a copy of the file to self.directory/clean by default.
        """
        raw_filename = f"FPL xPts forecast GW{gameweek} raw data.csv"
        filepath = os.path.join(self.directory, "raw", raw_filename)
        raw_data = pd.read_csv(filepath, encoding="utf-8")
        rows = raw_data.shape[0]
        
        extract_idx = lambda start_idx: [i for i in range(start_idx, rows+1, 4) if i <= rows]
        player_names = raw_data.loc[extract_idx(0), :]["col1"].dropna().reset_index(drop=True)
        positions_and_costs = raw_data.loc[extract_idx(1), :]["col1"].dropna().reset_index(drop=True)
        proj_exp_points = raw_data.loc[extract_idx(2), :].drop(columns=["col5", "col6", "col7", "col8"]).reset_index(drop=True)
        
        df = pd.concat([player_names, positions_and_costs, proj_exp_points], ignore_index=True, axis=1)
        df.columns = ["name", "position_and_costs", "xmins"] + [f"ep_gw{i}" for i in range(gameweek, gameweek + 3)]
        df[["position", "cost"]] = df["position_and_costs"].str.split(" ", expand=True)
        df = df[["name", "position", "cost", "xmins"] + [f"ep_gw{i}" for i in range(gameweek, gameweek + 3)]]
        df["position"] = df["position"].map({"GK": "GKP", "DF": "DEF", "MD":"MID", "FW": "FWD"})
        df["cost"] = df["cost"].astype("float32")
        df["xmins"] = df["xmins"].astype("int32").fillna(0)
        df["cost"] = np.where(df["cost"] == 99.9, 0, df["cost"])

        for i in range(gameweek, gameweek + 3):
            df[f"ep_gw{i}"] = df[f"ep_gw{i}"].fillna(0)
        
        # Data enrichment
        df["name"] = df["name"].replace(fpl_xPts_forecast_name_map)
        df["name"] = df.apply(lambda row: fpl_xPts_forecast_name_pos_map.get((row["name"], row["position"]), row["name"]), axis=1)
       
        if save_to_disk:
            filename = f"FPL {self.season_label} season - xPts forecast GW{gameweek} data.csv"
            df.to_csv(os.path.join(self.directory, "clean", filename) ,index=False)
        
        self.gw_forecast_df = df.copy()
        return df