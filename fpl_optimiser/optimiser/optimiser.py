import pandas as pd
import numpy as np
import pulp

class MultiGWTeamOptimizer:
    """
    """
    def __init__(self, 
                 player_data_df: pd.DataFrame, 
                 start_gameweek: int,
                 gameweeks: int = 3,
                 bench_weight: float = 0.5,
                 gkp_bench_weight: float = 0.1,
                 time_decay: float = 1.0,
                 max_price_change: float = 0.3,
                 k: float = 0.3,
                 use_price_model: bool = True,
                 validation: bool = True,
                 use_existing_team: bool = False) -> None:
        
        # Understand the data being fed in - if gw field exceeds self.gameweeks raise error.
        # Set attributes.
        self.player_data_df = player_data_df
        self.start_gameweek = start_gameweek
        self.gameweeks = gameweeks
        self.bench_weight = bench_weight
        self.gkp_bench_weight = gkp_bench_weight
        self.time_decay = time_decay
        self.max_price_change = max_price_change
        self.k = k
        self.use_price_model = use_price_model
        self.validation = validation
        self.use_existing_team = use_existing_team

        self.positions = ["GKP", "DEF", "MID", "FWD"]
        self.teams = ['ARS', 'AST', 'BOU', 'BRE', 'BRI', 'CHE', 'CRY', 'EVE', 'FUL', 'IPS',
                      'LEI', 'LIV', 'MCI', 'MUN', 'NEW', 'NOT', 'SOU', 'TOT', 'WHM', 'WOL']
        self.allowed_formations = [
            [3, 4, 3], [3, 5, 2], [4, 4, 2], [4, 3, 3],
            [4, 5, 1], [5, 3, 2], [5, 4, 1], [5, 2, 3]
        ]

    def sigmoid(self, x, k=0.3, midpoint=0) -> float:
        """ Sigmoid function to model non-linear player price adjustments."""
        return 1 / (1 + np.exp(-k * (x - midpoint)))

    def calc_optimal_team(self) -> pd.DataFrame:
        """Formulates and solves the ILP problem to select the optimal team over multiple gameweeks."""
        positions, teams = self.positions, self.teams
        start_gameweek, gameweeks, time_decay = self.start_gameweek, self.gameweeks, self.time_decay
        allowed_formations = self.allowed_formations
        player_data_df, k, use_existing_team = self.player_data_df, self.k, self.use_existing_team

        indices = player_data_df.index
        formation_dict = {i: formation for i, formation in enumerate(allowed_formations)}
        position_groups = {pos: set(player_data_df[player_data_df["position"] == pos].index) for pos in positions}
        team_groups = {team: set(player_data_df[player_data_df["team"] == team].index) for team in teams}

        if use_existing_team:
            if start_gameweek == 1:
                raise RuntimeError("You cannot have an existing team prior to GW1!")

            existing_team_df = pd.read_csv(f"FPL 24_25 season - team GW{start_gameweek - 1}.csv").drop(columns="position")
            player_data_df = pd.merge(player_data_df, existing_team_df, left_on="id", right_on="element", how="left")
            existing_team_indices = player_data_df[~player_data_df["element"].isna()]

            start_t, end_t = start_gameweek - 1, start_gameweek + gameweeks
        else:
            start_t, end_t = start_gameweek, start_gameweek + gameweeks

        # Initialize expected points and costs per gameweek
        pts_by_gw = {t: dict(zip(indices, np.array(player_data_df[f"ep_gw{t}"]) * (time_decay ** (t - 1)))) for t in range(start_gameweek, end_t)}
        baseline_pts_by_player = {i: player_data_df.at[i, f"ep_gw{start_gameweek}"] for i in indices}
        mins_played = dict(zip(indices, list(player_data_df["xmins"])))

        # Define estimated costs based on expected points and a sigmoid model for price changes
        estimated_costs_by_gw = self.estimate_costs(player_data_df, start_gameweek, end_t, pts_by_gw, baseline_pts_by_player)

        # Initialize linear programming problem
        prob = pulp.LpProblem("MaximizeObjectiveMultiGW", pulp.LpMaximize)

        # Define decision variables and constraints
        x_outfield, x_bench, x_captain, x_vice_captain, y_transfer_in, y_transfer_out, formation_vars = self.initialize_decision_vars(indices, start_t, end_t)

        # Add constraints and objective function
        self.add_constraints(prob, player_data_df, indices, teams, start_t, end_t, estimated_costs_by_gw, mins_played,
                             x_outfield, x_bench, x_captain, x_vice_captain, formation_vars, position_groups, team_groups)

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract the results and validation
        return self.extract_results(prob, start_t, end_t, player_data_df, x_outfield, x_bench, x_captain, x_vice_captain)

    def estimate_costs(self, player_data_df: pd.DataFrame, start_gameweek: int, end_t: int, pts_by_gw, baseline_pts_by_player) -> dict:
        """Estimates player costs using a sigmoid function for price change."""
        estimated_costs_by_gw = {}
        for t in range(start_gameweek, end_t):
            estimated_costs_by_gw[t] = {}
            for i in player_data_df.index:
                if t == start_gameweek or not self.use_price_model:
                    estimated_costs_by_gw[t][i] = player_data_df.at[i, "now_cost"]
                else:
                    score_diff = pts_by_gw[t][i] - baseline_pts_by_player[i]
                    price_change = self.max_price_change * (2 * self.sigmoid(score_diff) - 1)
                    estimated_costs_by_gw[t][i] = round(estimated_costs_by_gw[t-1][i] + price_change, 1)
        return estimated_costs_by_gw

    def initialize_decision_vars(self, indices, start_t, end_t):
        """Initializes decision variables."""
        x_outfield = pulp.LpVariable.dicts("x_outfield", (indices, range(start_t, end_t)), cat=pulp.LpBinary)
        x_bench = pulp.LpVariable.dicts("x_bench", (indices, range(start_t, end_t)), cat=pulp.LpBinary)
        x_captain = pulp.LpVariable.dicts("x_captain", (indices, range(start_t, end_t)), cat=pulp.LpBinary)
        x_vice_captain = pulp.LpVariable.dicts("x_vice_captain", (indices, range(start_t, end_t)), cat=pulp.LpBinary)
        y_transfer_in = pulp.LpVariable.dicts("y_transfer_in", (indices, range(start_t+1, end_t)), cat=pulp.LpBinary)
        y_transfer_out = pulp.LpVariable.dicts("y_transfer_out", (indices, range(start_t+1, end_t)), cat=pulp.LpBinary)
        formation_vars = pulp.LpVariable.dicts("formation_vars", (range(len(self.allowed_formations)), range(start_t, end_t)), cat=pulp.LpBinary)
        return x_outfield, x_bench, x_captain, x_vice_captain, y_transfer_in, y_transfer_out, formation_vars

    def add_constraints(self, prob, player_data_df, indices, teams, start_t, end_t, estimated_costs_by_gw, mins_played,
                        x_outfield, x_bench, x_captain, x_vice_captain, formation_vars, position_groups, team_groups):
        """Adds constraints and the objective function to the LP problem."""
        for t in range(self.start_gameweek, end_t):
            # Add various constraints (budget, players, formation, etc.)
            # Example:
            prob += pulp.lpSum([estimated_costs_by_gw[t][i] * (x_outfield[i][t] + x_bench[i][t]) for i in indices]) <= 100.0, f"BudgetConstraint_GW{t}"
            # Add more constraints as per the original problem...

    def extract_results(self, prob, start_t, end_t, player_data_df, x_outfield, x_bench, x_captain, x_vice_captain):
        """Extracts the solution and constructs the result dataframe."""
        results_df = pd.DataFrame()
        # Extract the results into a dataframe similar to the original code
        # Example:
        for t in range(start_t, end_t):
            outfield_indices = [i for i in player_data_df.index if pulp.value(x_outfield[i][t]) == 1]
            solution_df = player_data_df.loc[outfield_indices].copy()
            solution_df["gameweek"] = t
            results_df = pd.concat([results_df, solution_df], axis=0)
        return results_df
