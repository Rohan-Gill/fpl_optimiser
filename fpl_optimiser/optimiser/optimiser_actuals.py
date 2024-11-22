import pandas as pd
import numpy as np
import pulp
import time
import os

from ..utils import DATA_DIR

class MILPActualsOptimiser:
    """
    """

    POSITIONS = ["GKP", "DEF", "MID", "FWD"]
    TEAMS = ['ARS', 'AST', 'BOU', 'BRE', 'BRI', 'CHE', 'CRY', 'EVE', 'FUL', 'IPS',
             'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'NOT', 'SOU', 'TOT', 'WHM', 'WOL']

    # Define allowed formations for outfield players (Defenders, Midfielders, Forwards).
    FORMATIONS = [
        [3, 4, 3], [3, 5, 2], [4, 4, 2], [4, 3, 3],
        [4, 5, 1], [5, 3, 2], [5, 4, 1], [5, 2, 3]
    ]

    FORMATIONS_DICT = {idx: formation for idx, formation in enumerate(FORMATIONS)}  # Convert allowed formations in to a dictionary for easier indexing

    def __init__(self, 
                 player_data_df: pd.DataFrame, 
                 start_gameweek: int,
                 gameweeks: int = 3,
                 bench_weight: float = 0.5,
                 gkp_bench_weight: float = 0.1,                
                 validation: bool = True,
                 use_existing_team: bool = False) -> None:
        
        # Understand the data being fed in - if gw field exceeds self.gameweeks raise error.
        # Set attributes.
        self.player_data_df = player_data_df
        self.indices = player_data_df.index
        self.start_gameweek = start_gameweek
        self.gameweeks = gameweeks
        self.bench_weight = bench_weight
        self.gkp_bench_weight = gkp_bench_weight
        self.validation = validation
        self.use_existing_team = use_existing_team
        
        if use_existing_team:
            if start_gameweek ==  1:
                raise RuntimeError("Error: Cannot have an existing team prior to GW1")
            
            # Define dict to hold existing team dataframe index values.
            self.existing_team = {}
            filename = f"FPL 24_25 season - team GW{start_gameweek-1}.csv"
            existing_team_df = pd.read_csv(os.path.join(DATA_DIR, "official_api_data", filename)).drop(columns="position")
            player_data_df = pd.merge(player_data_df, existing_team_df, left_on="id", right_on="element", how="left")
            existing_team_indices = player_data_df[~player_data_df["element"].isna()]
            self.existing_team["outfield"] = set(existing_team_indices[existing_team_indices["multiplier"] != 0.0].index)
            self.existing_team["bench"] = set(existing_team_indices[existing_team_indices["multiplier"] == 0.0].index)
            self.existing_team["captain"] = set(existing_team_indices[existing_team_indices["is_captain"] == True].index)
            self.existing_team["vice_captain"] = set(existing_team_indices[existing_team_indices["is_vice_captain"] == True].index)

            # Define start and end points for temporal constraints
            # E.g. GW2 with existing team yields a projecton from GW1 -> GW4 inclusive.
            self.start_t, self.end_t = start_gameweek - 1, start_gameweek + gameweeks
        else:
            # Different start and end points for temporal constraints.
            self.start_t, self.end_t = start_gameweek, start_gameweek + gameweeks
        
        self.position_groups = {pos: {t: set(player_data_df[player_data_df[f"position_gw{t}"] == pos].index) for t in range(self.start_gameweek, self.end_t)} for pos in self.POSITIONS}
        self.team_groups = {team: {t: set(player_data_df[player_data_df[f"team_gw{t}"] == team].index) for t in range(self.start_gameweek, self.end_t)} for team in self.TEAMS}
        self.pts_by_gw = {t: dict(zip(self.indices, list(np.array(self.player_data_df[f"ep_gw{t}"])))) for t in range(self.start_gameweek, self.end_t)}
        self.mins_played = {t: dict(zip(self.indices, list(self.player_data_df[f"xmins_gw{t}"]))) for t in range(self.start_gameweek, self.end_t)}
        self.estimated_costs_by_gw = {t: dict(zip(self.indices, list(self.player_data_df[f"ep_cost_gw{t}"]))) for t in range(self.start_gameweek, self.end_t)}

    def objective_function(self) -> pulp.LpAffineExpression:
        """
        Defines the objective function to be used within the optimisation algorithm and returns a pulp.LpAffineExpression object.
        """
        # Objective function: Sum of expected points across all gameweeks.
        return pulp.lpSum([
            (self.pts_by_gw[t][idx] * self.x_captain[idx][t]) +  # Captain's points
            (self.pts_by_gw[t][idx] * (self.x_outfield[idx][t] + (self.gkp_bench_weight if self.player_data_df.at[idx, f"position_gw{t}"] == "GKP" else self.bench_weight) * self.x_bench[idx][t])) +
            (self.pts_by_gw[t][idx] * 0.1 * self.x_vice_captain[idx][t])  # Vice-captain's points
            for idx in self.indices for t in range(self.start_gameweek, self.end_t)
        ])

    def initialise_optimisation(self) -> None:
        """Initialise linear programming problem and define key decision variables."""
        
        # Create key decision variables.
        self.x_outfield = pulp.LpVariable.dicts("x_outfield", (self.indices, range(self.start_t, self.end_t)), cat=pulp.LpBinary)
        self.x_bench = pulp.LpVariable.dicts("x_bench", (self.indices, range(self.start_t, self.end_t)), cat=pulp.LpBinary)
        self.x_captain = pulp.LpVariable.dicts("x_captain", (self.indices, range(self.start_t, self.end_t)), cat=pulp.LpBinary)
        self.x_vice_captain = pulp.LpVariable.dicts("x_vice_captain", (self.indices, range(self.start_t, self.end_t)), cat=pulp.LpBinary)
        self.y_transfer_in = pulp.LpVariable.dicts("y_transfer_in", (self.indices, range(self.start_t + 1, self.end_t)), cat=pulp.LpBinary)
        self.y_transfer_out = pulp.LpVariable.dicts("y_transfer_out", (self.indices, range(self.start_t + 1, self.end_t)), cat=pulp.LpBinary)
        self.formation_vars = pulp.LpVariable.dicts("formation_vars", (range(len(self.FORMATIONS)), range(self.start_gameweek, self.end_t)), cat=pulp.LpBinary)
    
        # Initialise optimisation problem.
        self.prob = pulp.LpProblem("MaximizeObjectiveMultiGW", pulp.LpMaximize)  # create a pulp LpProblem and set it as an attribute.
        self.prob += self.objective_function(), "Objective"

    def add_constraints(self) -> None:
        """Adds objective function and constraints to the LP problem."""

        # If existing team provided then define GW-1 constraints.
        if self.use_existing_team and self.existing_team:
            # Equality constraints (set team selection).
            for idx in self.indices:
                if idx in self.existing_team["outfield"]:
                    self.prob += (self.x_outfield[idx][self.start_t] == 1, f"SetOutfieldValue_{idx}_GW{self.start_t}")
                if idx in self.existing_team["bench"]:
                    self.prob += (self.x_bench[idx][self.start_t] == 1, f"SetBenchValue_{idx}_GW{self.start_t}")
                if idx in self.existing_team["captain"]:
                    self.prob += (self.x_captain[idx][self.start_t] == 1, f"SetCaptainValue_{idx}_GW{self.start_t}")
                if idx in self.existing_team["vice_captain"]:
                    self.prob += (self.x_vice_captain[idx][self.start_t] == 1, f"SetViceCaptainValue_{idx}_GW{self.start_t}")
            
            # Inequality constraint - enforce team selection above.
            self.prob += pulp.lpSum([self.x_outfield[idx][self.start_t] for idx in self.indices]) == 11, f"OutfieldPlayersConstraint_GW{self.start_t}"
            self.prob += pulp.lpSum([self.x_bench[idx][self.start_t] for idx in self.indices]) == 4, f"BenchPlayersConstraint_GW{self.start_t}"

        # Base constraints
        for t in range(self.start_gameweek, self.end_t):
            self.prob += pulp.lpSum([self.estimated_costs_by_gw[t][idx] * (self.x_outfield[idx][t] + self.x_bench[idx][t]) for idx in self.indices]) <= 100.0, f"BudgetConstraint_GW{t}"            
            
            # Update starting constraint to reflect min_played over multiple gameweeks 
            #self.prob += pulp.lpSum([self.mins_played[t][idx] * (self.x_outfield[idx][t] + self.x_bench[idx][t]) for idx in self.indices]) >= 15 * 70.0, f"ProbabilityOfStartingConstraint_GW{t}"
            self.prob += pulp.lpSum([self.x_outfield[idx][t] for idx in self.indices]) == 11, f"OutfieldPlayersConstraint_GW{t}"
            self.prob += pulp.lpSum([self.x_bench[idx][t] for idx in self.indices]) == 4, f"BenchPlayersConstraint_GW{t}"

            # Update team constraints in-line with team_groups change
            for team in self.TEAMS:
                self.prob += pulp.lpSum([self.x_outfield[idx][t] + self.x_bench[idx][t] for idx in self.team_groups[team][t]]) <= 3, f"{team}TeamConstraint_GW{t}"
            
            for idx in self.indices:
                self.prob += pulp.lpSum([self.x_outfield[idx][t] + self.x_bench[idx][t]]) <= 1, f"SingleSelectionConstraint_GW{t}_{idx}"
                self.prob += self.x_captain[idx][t] <= self.x_outfield[idx][t], f"CaptainInOutfield_GW{t}_{idx}"
                self.prob += self.x_vice_captain[idx][t] <= self.x_outfield[idx][t], f"ViceCaptainInOutfield_GW{t}_{idx}"
                self.prob += self.x_captain[idx][t] + self.x_vice_captain[idx][t] <= 1, f"NotBothCaptainAndViceCaptain_GW{t}_{idx}"
            
            self.prob += pulp.lpSum([self.x_captain[idx][t] for idx in self.indices]) == 1, f"OneCaptain_GW{t}"
            self.prob += pulp.lpSum([self.x_vice_captain[idx][t] for idx in self.indices]) == 1, f"OneViceCaptain_GW{t}"    

            self.prob += pulp.lpSum([self.formation_vars[idx][t] for idx in self.FORMATIONS_DICT]) == 1, f"OneFormation_GW{t}"
            
            # Update position constraints in-line with position_groups change
            for form_idx, formation in self.FORMATIONS_DICT.items():
                num_def, num_mid, num_fwd = formation
                self.prob += pulp.lpSum([self.x_outfield[idx][t] for idx in self.position_groups["DEF"][t]]) >= num_def * self.formation_vars[form_idx][t], f"DefendersFormationOutfield_GW{t}_{form_idx}"
                self.prob += pulp.lpSum([self.x_outfield[idx][t] for idx in self.position_groups["MID"][t]]) >= num_mid * self.formation_vars[form_idx][t], f"MidfieldersFormationOutfield_GW{t}_{form_idx}"
                self.prob += pulp.lpSum([self.x_outfield[idx][t] for idx in self.position_groups["FWD"][t]]) >= num_fwd * self.formation_vars[form_idx][t], f"ForwardsFormationOutfield_GW{t}_{form_idx}"
            
            self.prob += pulp.lpSum([self.x_outfield[idx][t] for idx in self.position_groups["GKP"][t]]) == 1, f"GoalkeeperFormationOutfield_GW{t}"
            self.prob += pulp.lpSum([self.x_bench[idx][t] for idx in self.position_groups["GKP"][t]]) == 1, f"GoalkeeperFormationBench_GW{t}"
            self.prob += pulp.lpSum([self.x_outfield[idx][t] + self.x_bench[idx][t] for idx in self.position_groups["DEF"][t]]) == 5, f"DefendersLineupHardConstraint_GW{t}"
            self.prob += pulp.lpSum([self.x_outfield[idx][t] + self.x_bench[idx][t] for idx in self.position_groups["MID"][t]]) == 5, f"MidfieldersLineupHardConstraint_GW{t}"
            self.prob += pulp.lpSum([self.x_outfield[idx][t] + self.x_bench[idx][t] for idx in self.position_groups["FWD"][t]]) == 3, f"ForwardsLineupHardConstraint_GW{t}"   
            self.prob += pulp.lpSum([self.estimated_costs_by_gw[t][idx] * self.x_bench[idx][t] for idx in self.position_groups["GKP"][t]]) <= 4.0, f"BenchGK_CostLT4M_GW{t}"

        # Transfer constraints: At most one transfer in and out per gameweek
        for t in range(self.start_t + 1, self.end_t):
            self.prob += pulp.lpSum([self.y_transfer_out[idx][t] for idx in self.indices]) <= 1, f"MaxOneTransferOut_GW{t}"
            self.prob += pulp.lpSum([self.y_transfer_in[idx][t] for idx in self.indices]) <= 1, f"MaxOneTransferIn_GW{t}"
            
            for idx in self.indices:
                self.prob += self.y_transfer_out[idx][t] >= self.x_outfield[idx][t-1] + self.x_bench[idx][t-1] - self.x_outfield[idx][t] - self.x_bench[idx][t], f"TransferOutConsistency_GW{t}_{idx}"
                self.prob += self.y_transfer_in[idx][t] >= self.x_outfield[idx][t] + self.x_bench[idx][t] - self.x_outfield[idx][t-1] - self.x_bench[idx][t-1], f"TransferInConsistency_GW{t}_{idx}" 
    

    def extract_results(self) -> pd.DataFrame:
        """Extracts the solution and constructs a results dataframe representing the optimal team selection,
           which is assigned as an attribute of the class object.
        """

        results_df = pd.DataFrame()
        for t in range(self.start_t, self.end_t):
            outfield_indices = [idx for idx in self.indices if pulp.value(self.x_outfield[idx][t]) == 1]
            bench_indices = [idx for idx in self.indices if pulp.value(self.x_bench[idx][t]) == 1]
            captain_indices = [idx for idx in self.indices if pulp.value(self.x_captain[idx][t]) == 1]
            vice_captain_indices = [idx for idx in self.indices if pulp.value(self.x_vice_captain[idx][t]) == 1]
            
            solution_df = pd.concat([
                self.player_data_df.loc[outfield_indices, :].assign(position_type='Outfield'),
                self.player_data_df.loc[bench_indices, :].assign(position_type='Bench')
            ])
            
            # Add captain and vice-captain info
            solution_df["captain"] = solution_df.index.isin(captain_indices)
            solution_df["vice_captain"] = solution_df.index.isin(vice_captain_indices)

            # Define sorting variables
            solution_df["pos_rank"] = solution_df[f"position_gw{t}"].map(dict(zip(self.POSITIONS, range(0, len(self.POSITIONS)))))
            solution_df["pos_type_rank"] = solution_df["position_type"].map({"Outfield": 0 , "Bench": 1})
            solution_df.sort_values(by=["pos_type_rank", "pos_rank"], ascending=True, inplace=True)
            solution_df.drop(columns=["pos_rank", "pos_type_rank"], inplace=True)

            solution_df["gameweek"] = t
            
            if self.use_existing_team:
                solution_df[f"ep_cost_gw{self.start_t}"] = solution_df[f"ep_cost_gw{self.start_t + 1}"]
                solution_df[f"ep_gw{self.start_t}"] = solution_df[f"ep_gw{self.start_t + 1}"]
            
            solution_df = solution_df[["id", "name", f"position_gw{t}", f"team_gw{t}", f"prob_injury_gw{t}",
                                       f"xmins_gw{t}", f"ep_cost_gw{t}", "gameweek",
                                       f"ep_gw{t}", "position_type", "captain", "vice_captain"]]
            solution_df.rename(columns={f"position_gw{t}": "position", f"team_gw{t}": "team", 
                                       f"prob_injury_gw{t}": "prob_injury", f"ep_gw{t}":"xPts",
                                       f"ep_cost_gw{t}": "player_cost", f"xmins_gw{t}":"xMins"}, inplace=True)
            results_df = pd.concat([results_df, solution_df], axis=0)

            # Do not print validation report for first period if the solver is run with an existing team.
            if self.validation and not(self.use_existing_team and t == self.start_t):
                xPts_total_incl_cap = np.where(solution_df["position_type"] == "Outfield", 
                                            np.where(solution_df["captain"] == True,
                                                        2 * solution_df["xPts"],
                                                        solution_df["xPts"]
                                                    ), 
                                                    0
                                            ).sum()
                xPts_total_excl_cap = np.where(solution_df["position_type"] == "Outfield", solution_df["xPts"], 0).sum()
                formation_stats = dict(solution_df[solution_df["position_type"] == "Outfield"]["position"].value_counts())
                players_trns_out = set(results_df[results_df['gameweek'] == t-1]['name']) - set(results_df[results_df['gameweek'] == t]['name'])
                players_trns_in = set(results_df[results_df['gameweek'] == t]['name']) - set(results_df[results_df['gameweek'] == t-1]['name'])
                outfield_players_prev = set(results_df[(results_df['gameweek'] == t-1) & (results_df['position_type'] == 'Outfield')]['name'])
                outfield_players_curr = set(results_df[(results_df['gameweek'] == t) & (results_df['position_type'] == 'Outfield')]['name'])
                bench_players_prev = set(results_df[(results_df['gameweek'] == t-1) & (results_df['position_type'] == 'Bench')]['name'])
                bench_players_curr = set(results_df[(results_df['gameweek'] == t) & (results_df['position_type'] == 'Bench')]['name'])
                players_benched = (outfield_players_prev & bench_players_curr) - (players_trns_out | players_trns_in)
                players_promoted = (bench_players_prev & outfield_players_curr) - (players_trns_out | players_trns_in)

                print(f"Gameweek {t}:") 
                print(f"{pulp.LpStatus[self.prob.status]} team:\n{solution_df}\n")
                print(f"Formation: {formation_stats['DEF']},{formation_stats['MID']},{formation_stats['FWD']}")
                print(f"Total team cost: {solution_df['player_cost'].sum()}")
                print(f"   (o/w Outfield): {solution_df[solution_df['position_type'] == 'Outfield']['player_cost'].sum()}")
                print(f"   (o/w Bench): {solution_df[solution_df['position_type'] == 'Bench']['player_cost'].sum()}")
                print(f"Total expected points (excl. Captain): {xPts_total_excl_cap}")
                print(f"Total expected points (incl. Captain): {xPts_total_incl_cap}")
                print(f"Captain: {solution_df[solution_df['captain'] == True]['name'].values[0]}")
                print(f"Vice-Captain: {solution_df[solution_df['vice_captain'] == True]['name'].values[0]}")
                print(f"Transfered out: {'N/A' if t == (not self.use_existing_team and self.start_gameweek) else ''.join(players_trns_out)}")
                print(f"Transferred in: {'N/A' if t == (not self.use_existing_team and self.start_gameweek) else ''.join(players_trns_in)}")
                print(f"Players benched: {'N/A' if t == (not self.use_existing_team and self.start_gameweek) else ', '.join(players_benched)}")
                print(f"Players promoted: {'N/A' if t == (not self.use_existing_team and self.start_gameweek) else ', '.join(players_promoted)}\n")
        
        self.results_df = results_df

    def calulate_optimal_team(self) -> None:
        """ 
        Formulates and solves an LP problem that will calculate the optimal FPL team for a given gameweek, 
        based on a DataFrame including all FPL players for a given gameweek and a forecast of their projected
        points (xPts).
        """
        start_time = time.time()
        print(f"Calculating a {self.gameweeks}-gameweek forecast, starting from GW: {self.start_gameweek}...")
        self.initialise_optimisation()  # Create an LP problem and initialise key decision variables.
        self.add_constraints()  # Add objective function and constraint terms to the linear programming problem.

        # Solve the LP problem.
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results.
        self.extract_results()
        print("Optimisation process complete!")
        print(f"Time taken: {round(time.time() - start_time, 2)} seconds")