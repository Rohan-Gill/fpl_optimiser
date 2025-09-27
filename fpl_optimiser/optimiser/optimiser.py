import pandas as pd
import numpy as np
import pulp
import time
import os
import inspect
import requests

from typing import Optional
from ..utils import DATA_DIR, YAMLFile
from ..data import FplAPIData


class MILPOptimiser:
    """
    Class representing an Integer Linear Programming Optimiser using PuLP.
    This class should be used for future gameweek forecasting.

    The following must be provided when creating an object of this class:
        1). A Pandas DataFrame containing (player_data_df):
                - Forecasted expected points and minutes played
                - Each row must represent an unique player
                - Every player must have a position and team
                - Every player must have a cost as at the starting gameweek.  
        2). An integer representing the forecast starting point (start_gameweek).

    There are also optional keyword arguments that can be specified to enable
    a different optimisations and cater to different scenarios. 

    The class provides methods that should be run sequentially to set-up, solve
    and provide diagnostic reporting for the specified optimisation problem.

    Optimisation is performed over the future 3 gameweeks by default.
    """

    POSITIONS = ["GKP", "DEF", "MID", "FWD"]
    ALL_TEAMS = {
        "2024/2025": ['ARS', 'AST', 'BOU', 'BRE', 'BRI', 'CHE', 'CRY', 'EVE',
                      'FUL', 'IPS', 'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'NOT',
                      'SOU', 'TOT', 'WHM', 'WOL'],

        "2025/2026": ['ARS', 'AST', 'BUR', 'BOU', 'BRE', 'BRI', 'CHE', 'CRY',
                      'EVE', 'FUL', 'LEE', 'LIV', 'MCI', 'MUN', 'NEW', 'NOT',
                      'SUN', 'TOT', 'WHM', 'WOL']
    }

    # Define allowed formations for outfield players.
    FORMATIONS = [
        [3, 4, 3], [3, 5, 2], [4, 4, 2], [4, 3, 3],
        [4, 5, 1], [5, 3, 2], [5, 4, 1], [5, 2, 3]
    ]

    # Convert allowed formations in to a dictionary for easier indexing.
    FORMATIONS_DICT = {idx: formation for idx, formation in enumerate(FORMATIONS)}

    def __init__(self, 
                 player_data_df: pd.DataFrame, 
                 start_gameweek: int,
                 gameweeks: int = 3,
                 t0_team_value: float = 100.0,
                 excess_budget: float = 0.0,
                 free_transfers: int = 1,
                 bench_weight: float = 0.5,
                 gkp_bench_weight: float = 0.1,
                 time_decay: float = 1.0,
                 max_price_change: float = 0.3,
                 k: float = 0.3,
                 use_price_model: bool = True,
                 validation: bool = True,
                 use_existing_team: bool = False,
                 opt_rel_tol: float = 0.01,
                 config: Optional[YAMLFile] = None) -> None:

        # Set attributes.
        self.df = player_data_df
        self.indices = player_data_df.index
        self.start_gameweek = start_gameweek
        self.gameweeks = gameweeks
        self.t0_team_value = t0_team_value
        self.excess_budget = excess_budget
        self.free_transfers = free_transfers
        self.bench_weight = bench_weight
        self.gkp_bench_weight = gkp_bench_weight
        self.time_decay = time_decay
        self.max_price_change = max_price_change
        self.k = k
        self.use_price_model = use_price_model
        self.validation = validation
        self.use_existing_team = use_existing_team
        self.opt_rel_tol = opt_rel_tol
        self.config = config if config is not None else YAMLFile()
        self.teams = self.ALL_TEAMS[self.config.season]

        # Get default args
        self._budget_override = False
        sig = inspect.signature(self.__init__)
        results = {"t0_team_value": False, "excess_budget": False}
        for param_name, param in sig.parameters.items():
            if param_name in results.keys():      
                if getattr(self, param_name) != param.default:
                    results[param_name] = True

        # Provided budget is priority when running in existing team mode.
        if results["t0_team_value"] and results["excess_budget"]:
            self._budget_override = True

        # Error handling.
        if start_gameweek == 1 and t0_team_value + excess_budget > 100.0:
            raise RuntimeError("Error: Budget cannot be greater than £100mn in GW1!")

        # Configure existing team parameters.
        if use_existing_team:
            if start_gameweek ==  1:
                raise RuntimeError("Error: Existing team prior to GW1 not possible!")
            
            # Define dict to hold existing team dataframe index values.
            self.existing_team = {}
            s = self.config.season.replace("/", "_")
            s_num = "_".join([year[-2:] for year in s.split("_")])
            filename = f"{s}/FPL {s_num} season - team GW{start_gameweek - 1}.csv"
            temp = pd.read_csv(os.path.join(DATA_DIR, "official_api_data", filename))
            exist_team_df = pd.merge(player_data_df,
                                     temp,
                                     left_on="id", right_on="element",
                                     how="left",
                                     suffixes=("", "_right")
                                     )

            exist_team_df = exist_team_df[~exist_team_df["element"].isna()].copy()
            exist_team_outfield_df = exist_team_df[exist_team_df["multiplier"] != 0.0]
            exist_team_bench_df = exist_team_df[exist_team_df["multiplier"] == 0.0]
            exist_team_cap_df = exist_team_df[exist_team_df["is_captain"] == True]
            exist_team_vc_df = exist_team_df[exist_team_df["is_vice_captain"] == True]

            # Set indices
            self.existing_team["outfield"] = set(exist_team_outfield_df.index)
            self.existing_team["bench"] = set(exist_team_bench_df.index)
            self.existing_team["captain"] = set(exist_team_cap_df.index)
            self.existing_team["vice_captain"] = set(exist_team_vc_df.index)

            # Define start and end points for gameweek projection period constraints.
            # E.g. GW2 with existing team yields a projecton from GW1 -> GW4 inclusive.
            self.start_t, self.end_t = start_gameweek - 1, start_gameweek + gameweeks

            # If t0_team_value and excess_budget are not specified then extract. 
            if not self._budget_override:
                try:
                    api_data = FplAPIData()
                    squad, bank = api_data.get_team_value(gameweek=self.start_t)
                except requests.exceptions.RequestException as req_err:
                    print(f"Error: {req_err}")

                squad = exist_team_df["value"].sum() # values in CSV file more accurate.
                self.t0_team_value = squad
                self.excess_budget = bank
        else:
            # Different start and end points for temporal constraints.
            self.start_t, self.end_t = start_gameweek, start_gameweek + gameweeks

        # Set variables needed to define optimisation problem.
        self.position_groups = {
            p: set(self.df[self.df["position"] == p].index) for p in self.POSITIONS
        }

        self.team_groups = {
            t: set(self.df[self.df["team"] == t].index) for t in self.teams
        }

        self.pts_by_gw = {}
        self.mins_by_gw = {}
        for t in range(self.start_gameweek, self.end_t):
            # Expected points by gameweek with time decay applied.
            self.pts_by_gw[t] = dict(
                zip(
                    self.indices,
                    list(np.array(self.df[f"xpts_gw{t}"]) * (time_decay ** (t - 1)))
                    )
            )

            # Expected minutes to be played by gameweek.
            self.mins_by_gw[t] = dict(
                zip(
                    self.indices, 
                    list(self.df[f"xmins_gw{t}"])
                )
            )

        # Initial expected points (baseline for each player).
        # Can use self.pts_by_gw at start gameweek value for dict key?
        self.baseline_pts_by_gw = {
            i: self.df.at[i, f"xpts_gw{start_gameweek}"] for i in self.indices}

        # Calculate and add estimated player costs to dataframe.
        self.costs_by_gw = self.estimate_player_costs()
        for t in range(self.start_gameweek, self.end_t):
            self.df[f"xcost_gw{t}"] = self.df.index.map(lambda i: self.costs_by_gw[t][i])

    @property
    def budget_override(self):
        """ Read only property """
        return self._budget_override

    @staticmethod
    def sigmoid(x, k=0.3, midpoint=0) -> float:
        """ Sigmoid function used to model non-linear player price adjustments."""
        return 1 / (1 + np.exp(-k * (x - midpoint)))
    
    def estimate_player_costs(self) -> dict:
        """
        Creates a dict of player index and estimated player cost key-value pairs
        for each gameweek.
        Player costs are forecasted using a sigmoid function for price change.
        """
        costs_by_gw = {}
        for t in range(self.start_gameweek, self.end_t):
            costs_by_gw[t] = {}
            for i in self.indices:
                if t == self.start_gameweek or not self.use_price_model:
                    costs_by_gw[t][i] = self.df.at[i, "now_cost"]
                else:
                    # Apply Sigmoid function to shift in pts_by_gw (vs. start GW).
                    score_diff = self.pts_by_gw[t][i] - self.baseline_pts_by_gw[i]
                    cost_func = self.sigmoid(score_diff, k=self.k)

                    # Scale Sigmoid output by max price change observed historically.
                    price_change = self.max_price_change * (2 * cost_func - 1)
                    costs_by_gw[t][i] = round(costs_by_gw[t-1][i] + price_change, 1)
        
        return costs_by_gw

    def objective_function(self) -> pulp.LpAffineExpression:
        """
        Defines the objective function to be used in the optimisation.
        Returns a pulp.LpAffineExpression as per PuLP MILP solver requirements.
        """
        # Sum of expected points (with time decay applied) across all gameweeks.
        # Points deductions for any transfers above the total available are applied. 
        return pulp.lpSum([
            (self.pts_by_gw[t][i] * self.x_captain[i][t]) +
            (self.pts_by_gw[t][i] * self.x_outfield[i][t]) +
            (
                self.pts_by_gw[t][i] * self.x_bench[i][t] *
                (
                    self.gkp_bench_weight 
                    if self.df.at[i, "position"] == "GKP"
                    else self.bench_weight
                )
            ) +
            (self.pts_by_gw[t][i] * 0.1 * self.x_vice_captain[i][t])
            for i in self.indices for t in range(self.start_gameweek, self.end_t)
        ] + [self.transfers_over[t] * -4 for t in range(self.start_t + 1, self.end_t)])

    def initialise_optimisation(self) -> None:
        """
        Defines a PuLP LP (maximisation) problem and adds the objective function
        to it. The necessary PuLP binary decision variables are also defined.
        """

        # Define optimisation problem.
        self.prob = pulp.LpProblem("MaximizeObjectiveMultiGW", pulp.LpMaximize)
        
        # Define key decision variables.
        decision_vars = ["x_outfield", "x_bench", "x_captain", "x_vice_captain", 
                         "y_transfer_in", "y_transfer_out", "formation",
                         "team_value", "bank", "budget",
                         "transfers_made", "transfers_over", "transfers_available"]
        
        f = lambda x, y, z: pulp.LpVariable.dicts(name=x, indices=y, cat=z)
        for var in decision_vars:
            cat = pulp.LpBinary
            if var in ["y_transfer_in", "y_transfer_out"]:
                name_indices = self.indices
                gameweek_indices = range(self.start_t + 1, self.end_t)
                indices = (name_indices, gameweek_indices)
            elif var == "formation":
                name_indices = range(len(self.FORMATIONS))
                gameweek_indices = range(self.start_gameweek, self.end_t)
                indices = (name_indices, gameweek_indices)
            elif var in ["team_value", "bank", "budget"]:
                cat = pulp.LpContinuous
                indices = range(self.start_gameweek, self.end_t)
            elif var in ["transfers_made", "transfers_over", "transfers_available"]:
                # Variables hold values as at end of current GW.
                cat = pulp.LpInteger
                indices = range(self.start_t + 1, self.end_t)
            else:
                name_indices = self.indices
                gameweek_indices = range(self.start_t, self.end_t)
                indices = (name_indices, gameweek_indices)

            setattr(self, var, f(var, indices, cat))
        
        # Add objective function to LP problem.
        self.prob += self.objective_function(), "Objective"

    def add_constraints(self) -> None:
        """Defines and constraints to the PuLP LP problem."""

        # Existing team constraints (GW-1 constraints).
        if self.use_existing_team and self.existing_team:
            # Equality constraints (set team selection).
            # Set decision variables to 1 based on existing team data.
            # Cannot set decision variables in PuLP so enforce above via contraints.
            for role in ["outfield", "bench", "captain", "vice_captain"]:
                xvar = getattr(self, f"x_{role}")
                for i in self.indices:
                    role_fstr = role.title().replace("_", "")
                    cons_name = f"Set{role_fstr}Value_{i}_GW{self.start_t}"
                    selected = 1 if i in self.existing_team[role] else 0
                    self.prob += (xvar[i][self.start_t] == selected, cons_name)
            
            # Enforce team selection above.
            # Should be 11 outfield and 4 bench players.
            for role in ["outfield", "bench"]:
                xvar = getattr(self, f"x_{role}")
                num_selected = [xvar[i][self.start_t] for i in self.indices]
                selected_limit = 11 if role == "outfield" else 4
                cons_name = f"{role.title()}Players_GW{self.start_t}"
                self.prob += (pulp.lpSum(num_selected) == selected_limit, cons_name)

        # Gameweek forecast constraints.
        for t in range(self.start_gameweek, self.end_t):
            xb = lambda i: self.x_outfield[i][t] + self.x_bench[i][t]
            xb_prev = lambda i: self.x_outfield[i][t-1] + self.x_bench[i][t-1]

            # Budget, player starting probability, 11 outfield and 4 bench players.            
            tot_cost = [self.costs_by_gw[t][i] * xb(i) for i in self.indices]
            tot_xmins = [self.mins_by_gw[t][i] * xb(i) for i in self.indices]
            out_selected = [self.x_outfield[i][t] for i in self.indices]
            bench_selected = [self.x_bench[i][t] for i in self.indices]

            t0_budget = self.t0_team_value + self.excess_budget
            if t == self.start_gameweek:
                t0_bank = t0_budget - pulp.lpSum(tot_cost)
                self.prob += (self.bank[t] == t0_bank, f"SetBankValue_GW{t}")
                self.prob += (self.budget[t] == t0_budget, f"SetBudget_GW{t}")
            else:
                gained = [self.costs_by_gw[t][i] * self.y_transfer_out[i][t]
                          for i in self.indices]
                lost = [self.costs_by_gw[t][i] * self.y_transfer_in[i][t]
                        for i in self.indices]
                curr_bank_val = self.bank[t-1] + pulp.lpSum(gained) - pulp.lpSum(lost)
                self.prob += (self.bank[t] == curr_bank_val, f"SetBankValue_GW{t}")

                curr_team_val = [self.costs_by_gw[t][i] * xb_prev(i) for i in self.indices]
                curr_budget_val = pulp.lpSum(curr_team_val) + self.bank[t-1]
                self.prob += (self.budget[t] == curr_budget_val, f"SetBudget_GW{t}")

            self.prob += (self.team_value[t] == pulp.lpSum(tot_cost), f"SetTeamValue_GW{t}")
            self.prob += (self.bank[t] >= 0, f"BankFloor_GW{t}")
            self.prob += (pulp.lpSum(tot_cost) <= self.budget[t], f"Budget_GW{t}")
            self.prob += (pulp.lpSum(tot_xmins) >= 15 * 70.0, f"StartingProb_GW{t}")
            self.prob += (pulp.lpSum(out_selected) == 11, f"OutfieldPlayers_GW{t}")
            self.prob += (pulp.lpSum(bench_selected) == 4, f"BenchPlayers_GW{t}")
            
            # Ensure only 3 players are selected from a given team.
            for team in self.teams:
                team_t = [xb(i) for i in self.team_groups[team]]
                self.prob += (pulp.lpSum(team_t) <= 3, f"{team}Team_GW{t}")
            
            # Select a single captain and vice captain from outfield players selected.
            # A single outfield player cannot be both captain and vice captain.
            for i in self.indices:
                single_selection_t = pulp.lpSum(xb(i)) <= 1
                cap_outfield = self.x_captain[i][t] <= self.x_outfield[i][t]
                vcap_outfield = self.x_vice_captain[i][t] <= self.x_outfield[i][t]
                one_cap_vcap = self.x_captain[i][t] + self.x_vice_captain[i][t] <= 1

                self.prob += (single_selection_t, f"SingleSelection_GW{t}_{i}")
                self.prob += (cap_outfield, f"CaptainInOutfield_GW{t}_{i}")
                self.prob += (vcap_outfield, f"ViceCaptainInOutfield_GW{t}_{i}")
                self.prob += (one_cap_vcap, f"NotBothCaptainAndViceCaptain_GW{t}_{i}")
            
            cap_selected = [self.x_captain[i][t] for i in self.indices]
            vcap_selected = [self.x_vice_captain[i][t] for i in self.indices]
            self.prob += (pulp.lpSum(cap_selected) == 1, f"OneCaptain_GW{t}")
            self.prob += (pulp.lpSum(vcap_selected) == 1, f"OneViceCaptain_GW{t}")    

            # Select a single formation.
            forms_selected = [self.formation[i][t] for i in self.FORMATIONS_DICT]
            self.prob += (pulp.lpSum(forms_selected) ==  1, f"OneFormation_GW{t}")

            # Number of selected players in each position must align to formation.
            for i, formation in self.FORMATIONS_DICT.items():
                for pos, num in zip(["DEF", "MID", "FWD"], formation):
                    out_pos_plys = [self.x_outfield[j][t] for j in self.position_groups[pos]]
                    tot_out_plys_in_pos = pulp.lpSum(out_pos_plys)
                    cname = f"Formation{pos}s_GW{t}_{i}"
                    self.prob += (tot_out_plys_in_pos >= num * self.formation[i][t], cname)
            
            # Only a single outfield keeper and bench keeper is allowed.
            out_gks = [self.x_outfield[i][t] for i in self.position_groups["GKP"]]
            bench_gks = [self.x_bench[i][t] for i in self.position_groups["GKP"]]
            self.prob += (pulp.lpSum(out_gks) ==  1, f"OneGKPOutfield_GW{t}")
            self.prob += (pulp.lpSum(bench_gks) == 1, f"OneGKPBench_GW{t}")

            # A maximum of 5 defenders, 5 midfielders and 3 forwards are allowed.
            for pos in ["DEF", "MID", "FWD"]:
                tot_plys_in_pos = [xb(i) for i in self.position_groups[pos]]
                max_num = 3 if pos == "FWD" else 5
                cname = f"Max{pos}sAllowedInFormation_GW{t}"
                self.prob += (pulp.lpSum(tot_plys_in_pos) == max_num, cname)
        
        # Transfer constraints.
        for t in range(self.start_t + 1, self.end_t):
            self.prob += (self.transfers_made[t] >= 0, f"TransfersMadeFloor_GW{t}")
            self.prob += (self.transfers_over[t] >= 0, f"TransfersOverFloor_GW{t}")
            self.prob += (self.transfers_available[t] >= 0, f"TransfersAvailFloor_GW{t}")

            plys_trans_out = [self.y_transfer_out[i][t] for i in self.indices]
            transfers_out = self.transfers_made[t] == pulp.lpSum(plys_trans_out)
            self.prob += (transfers_out, f"TransfersMade_GW{t}")

            if t == self.start_t + 1:
                avail = self.free_transfers - self.transfers_made[t]
                made = self.transfers_made[t] - self.free_transfers
            else:
                avail = 1 + self.transfers_available[t-1] - self.transfers_made[t]
                made = self.transfers_made[t] - (1 + self.transfers_available[t-1])
            
            self.prob += (self.transfers_available[t] == avail, f"TransfersAvail_GW{t}")
            self.prob += (self.transfers_over[t] == made, f"TransfersOver_GW{t}")

            for i in self.indices:
                x_prev = self.x_outfield[i][t-1] + self.x_bench[i][t-1]
                x_curr = self.x_outfield[i][t] + self.x_bench[i][t]

                # Exact change equation.
                chg = x_curr == x_prev + self.y_transfer_in[i][t] - self.y_transfer_out[i][t]
                self.prob += (chg, f"ChangeEq_GW{t}_{i}")

                # Logical bounds (no ghost transfers).
                no_trns_in = self.y_transfer_in[i][t] <= 1 - x_prev
                no_trns_out = self.y_transfer_out[i][t] <= x_prev
                no_trns_both = self.y_transfer_in[i][t] + self.y_transfer_out[i][t] <= 1
                self.prob += (no_trns_in, f"NoTrnsIn_GW{t}_{i}")
                self.prob += (no_trns_out, f"NoTrnsOut_GW{t}_{i}")
                self.prob += (no_trns_both, f"NoTrnsBoth_GW{t}_{i}")

    def extract_results(self) -> pd.DataFrame:
        """
        Extracts the solution of the PuLP LP problem by referencing the decision variables
        which are attributes of the MILPOptimiser class. Constructs a Pandas DataFrame object
        containing the optimal team selection for each gameweek.
        This method also prints a summary report for each gameweek by default. 
        """

        results_df = pd.DataFrame()
        for t in range(self.start_t, self.end_t):
            out_idx = [i for i in self.indices if pulp.value(self.x_outfield[i][t]) == 1]
            bench_idx = [i for i in self.indices if pulp.value(self.x_bench[i][t]) == 1]
            cap_idx = [i for i in self.indices if pulp.value(self.x_captain[i][t]) == 1]
            vcap_idx = [i for i in self.indices if pulp.value(self.x_vice_captain[i][t]) == 1]
            
            solution_df = pd.concat([
                self.df.loc[out_idx, :].assign(position_type="Outfield"),
                self.df.loc[bench_idx, :].assign(position_type="Bench")
            ])
            
            # Add captain and vice-captain info.
            solution_df["captain"] = solution_df.index.isin(cap_idx)
            solution_df["vice_captain"] = solution_df.index.isin(vcap_idx)

            # Define sorting variables.
            sort_vars = ["pos_type_rank", "pos_rank"]
            pos_map = {pos: idx for idx, pos in enumerate(self.POSITIONS)}
            pos_type_map = {"Outfield": 0 , "Bench": 1}

            solution_df["pos_rank"] = solution_df["position"].map(pos_map)
            solution_df["pos_type_rank"] = solution_df["position_type"].map(pos_type_map)
            solution_df.sort_values(by=sort_vars, ascending=True, inplace=True)
            solution_df.drop(columns=sort_vars, inplace=True)

            solution_df["gameweek"] = t

            if self.use_existing_team:
                for c in ["xmins", "xcost", "xpts"]:
                    col_t, col_t_plus1 = f"{c}_gw{self.start_t}", f"{c}_gw{self.start_t + 1}"
                    solution_df[col_t] = solution_df[col_t_plus1]
            
            solution_df = solution_df[["id", "name", "position", "team", "prob_injury",
                                       "starts", "starts_perc", "selected_by_percent",
                                       f"xmins_gw{t}", f"xcost_gw{t}", "gameweek",
                                       f"xpts_gw{t}", "position_type", "captain",
                                       "vice_captain"]]
            solution_df.rename(columns={f"xpts_gw{t}": "xPts",
                                        f"xcost_gw{t}": "player_cost",
                                        f"xmins_gw{t}": "xMins"
                                        }, inplace=True)
            results_df = pd.concat([results_df, solution_df], axis=0)

            # Do not report for 1st period if the solver is run in existing team mode.
            if self.validation and not(self.use_existing_team and t == self.start_t):
                curr_out = solution_df["position_type"] == "Outfield"
                curr_bench = solution_df["position_type"] == "Bench"
                curr_cap = solution_df["captain"] == True
                curr_vcap = solution_df["vice_captain"] == True

                # Can't report transfers/benchings if in the 1st period and no existing team.
                skip = not self.use_existing_team and t == self.start_gameweek
                xpts_sum_no_cap = np.where(curr_out, solution_df["xPts"], 0).sum()
                xpts_sum_cap = np.where(curr_out, 
                                        np.where(curr_cap,
                                                 2 * solution_df["xPts"], 
                                                 solution_df["xPts"]
                                                 ),
                                        0
                                        ).sum()

                formation_stats = dict(solution_df[curr_out]["position"].value_counts())
                formation = ",".join([str(formation_stats[p]) for p in ["DEF", "MID", "FWD"]])
                prev_gw = results_df["gameweek"] == t-1
                curr_gw = results_df["gameweek"] == t
                all_out = results_df["position_type"] == "Outfield"
                all_bench = results_df["position_type"] == "Bench"

                all_plys_prev = set(results_df[prev_gw]["name"])
                all_plys_curr = set(results_df[curr_gw]["name"])
                out_plys_prev = set(results_df[(prev_gw) & (all_out)]["name"])
                out_plys_curr = set(results_df[(curr_gw) & (all_out)]["name"])
                bench_plys_prev = set(results_df[(prev_gw) & (all_bench)]["name"])
                bench_plys_curr = set(results_df[(curr_gw) & (all_bench)]["name"])

                plys_trns_out = all_plys_prev - all_plys_curr
                plys_trns_in = all_plys_curr - all_plys_prev
                plys_trns_out_and_in = (plys_trns_out | plys_trns_in)
                plys_benched = (out_plys_prev & bench_plys_curr) - plys_trns_out_and_in
                plys_promoted = (bench_plys_prev & out_plys_curr) - plys_trns_out_and_in
                fdisplay = lambda obj: "".join(obj) if len(obj) == 1 else ", ".join(obj)

                print(f"Gameweek {t}:") 
                print(f"{pulp.LpStatus[self.prob.status]} team:\n{solution_df}\n")
                print(f"Formation: {formation}")
                print(f"Total budget: {pulp.value(self.budget[t])}")
                print(f" o/w Funds in bank: {pulp.value(self.bank[t])}")
                print(f" o/w Team cost: {round(solution_df['player_cost'].sum(),1)}")
                print(f"   o/w Outfield: {round(solution_df[curr_out]['player_cost'].sum(),1)}")
                print(f"   o/w Bench: {round(solution_df[curr_bench]['player_cost'].sum(),1)}")
                print(f"Total expected points (excl. Captain): {round(xpts_sum_no_cap,1)}")
                print(f"Total expected points (incl. Captain): {round(xpts_sum_cap,1)}")
                print(f"Captain: {solution_df[curr_cap]['name'].values[0]}")
                print(f"Vice-Captain: {solution_df[curr_vcap]['name'].values[0]}")
                print(f"Transfered out: {'N/A' if skip else fdisplay(plys_trns_out)}")
                print(f"Transferred in: {'N/A' if skip else fdisplay(plys_trns_in)}")
                print(f"Players benched: {'N/A' if skip else fdisplay(plys_benched)}")
                print(f"Players promoted: {'N/A' if skip else fdisplay(plys_promoted)}\n")
        
        self.results_df = results_df

    def calculate_optimal_team(self) -> None:
        """ 
        Formulates and solves a PuLP LP problem that will forecast 
        the optimal FPL team, from a given starting gameweek, based on an input
        Pandas DataFrame including forecast xMins and xPts data for all FPL players.
        """
        start_time = time.time()
        msg = (
            f"Calculating a {self.gameweeks}-gameweek forecast, "
            f"starting from GW: {self.start_gameweek}..."
        )
        print(msg)
        self.initialise_optimisation()
        self.add_constraints()

        # Solve the LP problem.
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=self.opt_rel_tol))
       
        # Extract results.
        self.extract_results()
        print("Optimisation process complete!")
        print(f"Time taken: {round(time.time() - start_time, 2)} seconds")

    def debug(self, player_name: str) -> None:
        """
        For the specified player, a diagnostic report is produced that prints the value
        of the decision variables and constraints in each possible gameweek.
        Team-level constraints are also shown.
        Individual team/formation decision variable values and constraints are not shown.
        """

        df = self.df[self.df["name"] == player_name]
        index, pos, team = df.index[0], df["position"].iat[0], df["team"].iat[0]

        # Print output report for decision variables.
        # C/VC decision variables only set for actual C/VC in existing team data!
        x_dvars = ["x_outfield", "x_bench", "x_captain", "x_vice_captain"]
        y_dvars = ["y_transfer_out", "y_transfer_in"]
        budget_dvars = ["team_value", "bank", "budget"]
        transfer_dvars = ["transfers_made", "transfers_over", "transfers_available"]

        print("\nDecision variables:")
        for dvar in x_dvars + y_dvars + budget_dvars + transfer_dvars:
            if dvar in x_dvars:
                start = self.start_t
            elif dvar in y_dvars + transfer_dvars:
                start = self.start_t + 1
            else:
                # For team budget variables.
                start = self.start_gameweek
            end = self.end_t
            dvar_val = getattr(self, dvar)[index]
            msg = [f"{dvar}_gw{i}: {pulp.value(dvar_val[i])}" for i in range(start, end)]
            print(", ".join(msg))         

        # Print output report for constraints.
        print("\nConstraints:")
        all_cons = self.prob.constraints
        def cons_summary(c_name: str, suppress: bool = False) -> str:
            cons = all_cons[c_name]
            cons_str = cons.asCplexLpConstraint(name=c_name).replace("\n", "")
            cons_lhs_val = cons.value() - cons.constant
            cons_rhs_val = -cons.constant
            cons_sense = cons.sense
            if cons_sense == 0:
                status = "SATISFIED" if cons_lhs_val == cons_rhs_val else "VIOLATED!"
            elif cons_sense == 1:
                status = "SATISFIED" if cons_lhs_val >= cons_rhs_val else "VIOLATED!"
            elif cons_sense == -1:
                status = "SATISFIED" if cons_lhs_val <= cons_rhs_val else "VIOLATED!"
            msg = f"{cons_str}" if not suppress else f"{c_name}"
            msg += f" LHS: {cons_lhs_val}, RHS: {cons_rhs_val}, {status}"
            return msg

        # GW-1 constraints if existing team mode has been activated.
        if self.use_existing_team and self.existing_team:
            for dvar in x_dvars:
                x_dvar_fstr = dvar[2:].title().replace("_", "")
                cons_name = f"Set{x_dvar_fstr}Value_{index}_GW{self.start_t}"
                msg = cons_summary(cons_name)
                print(msg)

            for dvar in ["x_outfield", "x_bench"]:
                cons_name = f"{dvar[2:].title()}Players_GW{self.start_t}"
                msg = cons_summary(cons_name, suppress=True)
                print(msg)

            print("")
        
        #  Gameweek forecast constraints.
        for i in range(self.start_gameweek, self.end_t):
            for cons in ["SetBankValue", "SetBudget", "SetTeamValue", "BankFloor",
                         "Budget", "StartingProb", "OutfieldPlayers", "BenchPlayers"]:
                cons_name = f"{cons}_GW{i}"
                suppress_option = False if cons == "BankFloor" else True
                msg = cons_summary(cons_name, suppress=suppress_option)
                print(msg)

            constraints = ["SingleSelection", "CaptainInOutfield", "CaptainInOutfield",
                           "ViceCaptainInOutfield", "NotBothCaptainAndViceCaptain"]
            for cons in constraints:
                cons_name = f"{cons}_GW{i}_{index}"
                msg = cons_summary(cons_name)
                print(msg)

            constraints = ["OneCaptain", "OneViceCaptain", "OneFormation",
                           "OneGKPOutfield", "OneGKPBench"]
            for cons in constraints:
                cons_name = f"{cons}_GW{i}"
                suppress_option = False if cons == "OneFormation" else True
                msg = cons_summary(cons_name, suppress=suppress_option)
                print(msg)
            
            print("")
            
        for i in range(self.start_t + 1, self.end_t):
            ply_level_cons = ["ChangeEq", "NoTrnsIn", "NoTrnsOut", "NoTrnsBoth"]
            team_level_cons = ["TransfersMadeFloor", "TransfersOverFloor",
                               "TransfersAvailFloor", 
                               "TransfersMade", "TranfersOver", "TransferAvail"]

            for cons in ply_level_cons + team_level_cons:
                if cons in ply_level_cons:
                    cons_name = f"{cons}_GW{i}_{index}" 
                else:
                    cons_name = f"{cons}_GW{i}"

                suppress_option = False if cons in ply_level_cons else True
                msg = cons_summary(cons_name, suppress=suppress_option)
                print(msg)
            
            print("")