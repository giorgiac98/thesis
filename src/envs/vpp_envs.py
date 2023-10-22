"""
    Single step and MDP version of the VPP environments.
"""
from datetime import datetime, timedelta

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation
import numpy as np
import pandas as pd
import random
from gurobipy import Model, GRB
import gurobipy
from matplotlib import pyplot as plt
from tabulate import tabulate
from typing import Tuple, List, Union, Any

########################################################################################################################

# Timesteps of 15 minutes in a day for the EMS use case
TIMESTEP_IN_A_DAY = 96


def instances_preprocessing(instances: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PV and Load values from string to float.
    :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
    :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
    """

    assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
    assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

    # Instances pv from file
    instances.loc[:, 'PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances.loc[:, 'PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    # Instances load from file
    instances.loc[:, 'Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances.loc[:, 'Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

    return instances


class VPPEnv(Env):
    """
    Gym environment for the VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render_modes": ["ascii"]
    }
    # Reward shaping due to constraints violation
    MIN_REWARD = -10000

    def __init__(self,
                 predictions,
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None,
                 logger=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """
        self.logger = logger
        self._do_log = False
        # Set numpy random seed to ensure reproducibility
        np.random.seed(0)

        # Number of timesteps in one day
        self.n = 96

        # Standard deviation of the additive gaussian noise
        self.noise_std_dev = noise_std_dev

        # These are variables related to the optimization model
        self.predictions = predictions
        self.predictions = instances_preprocessing(self.predictions)
        self.c_grid = c_grid
        self.shift = shift
        self.cap_max = 1000
        self.in_cap = 800
        self.c_diesel = 0.054
        self.p_diesel_max = 1200

        # We randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())

        self.savepath = savepath

        self._create_instance_variables()
        self._load_optimal_values()

    def do_log(self, do_log: bool):
        self._do_log = do_log

    def _load_optimal_values(self):
        data_dir = 'envs/ems_data/oracle'
        self.optimal_solution = pd.read_csv(f'{data_dir}/{self.mr}_solution.csv', index_col=0)
        self.optimal_cost = np.load(f'{data_dir}/{self.mr}_cost.npy')

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        # predicted PV for the current instance
        self.p_ren_pv_pred = self.predictions['PV(kW)'][self.mr]
        self.p_ren_pv_pred = np.asarray(self.p_ren_pv_pred)

        # predicted Load for the current instance
        self.tot_cons_pred = self.predictions['Load(kW)'][self.mr]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        # The real PV for the current instance is computed adding noise to the predictions
        noise = np.random.normal(0, self.noise_std_dev, self.n)
        self.p_ren_pv_real = self.p_ren_pv_pred + self.p_ren_pv_pred * noise

        # The real Load for the current instance is computed adding noise to the predictions
        noise = np.random.normal(0, self.noise_std_dev, self.n)
        self.tot_cons_real = self.tot_cons_pred + self.tot_cons_pred * noise

    def step(self, action: np.array):
        """
        Step function of the Gym environment.
        :param action: numpy.array; agent's action.
        :return:
        """
        raise NotImplementedError()

    def _get_observations(self):
        """
        :return: The observation of the specific environment implementation
        """
        raise NotImplementedError()

    def _clear(self):
        """
        Clear all the instance dependent variables.
        """
        raise NotImplementedError()

    def reset(self, seed=None, options=None) -> tuple[np.array, dict[str, Any]]:
        """
        When we reset the environment we randomly choose another instance and we clear all the instance variables.
        :return: numpy.array; pv and load values for the current instance.
        """
        super().reset(seed=seed)
        self._clear()

        # We randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())
        self._create_instance_variables()
        return self._get_observations(), {}

    def render(self, mode: str = 'ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = VPPEnv.timestamps_headers(self.n)
        print('\nPredicted PV(kW)')
        print(tabulate(np.expand_dims(self.p_ren_pv_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal PV(kW)')
        print(tabulate(np.expand_dims(self.p_ren_pv_real, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_real, axis=0), headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass

    @staticmethod
    def solve(mod: gurobipy.Model) -> bool:
        """
        Solve an optimization model.
        :param mod: gurobipy.Model; the optimization model to be solved.
        :return: bool; True if the optimal solution is found, False otherwise.
        """

        mod.setParam('OutputFlag', 0)
        mod.optimize()
        status = mod.status
        if status == GRB.Status.UNBOUNDED:
            print('\nThe model is unbounded')
            return False
        elif status == GRB.Status.INFEASIBLE:
            print('\nThe model is infeasible')
            return False
        elif status == GRB.Status.INF_OR_UNBD:
            print('\nThe model is either infeasible or unbounded')
            return False

        if status != GRB.Status.OPTIMAL:
            print('\nOptimization was stopped with status %d' % status)
            return False

        return True

    @staticmethod
    def timestamps_headers(num_timeunits: int) -> List[str]:
        """
        Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
        For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
        :param num_timeunits: int; the number of timeunits in a day.
        :return: list of string; list of timeunits.
        """

        start_time = datetime.strptime('00:00', '%H:%M')
        timeunit = 24 * 60 / num_timeunits
        timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
        timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

        return timestamps

    @staticmethod
    def min_max_scaler(starting_range: Union[Tuple[float, float], Tuple[int, int]],
                       new_range: Union[Tuple[float, float], Tuple[int, int]],
                       value: float) -> float:
        """
        Scale the input value from a starting range to a new one.
        :param starting_range: tuple of float; the starting range.
        :param new_range: tuple of float; the new range.
        :param value: float; value to be rescaled.
        :return: float; rescaled value.
        """

        assert isinstance(starting_range, tuple) and len(starting_range) == 2, \
            "feature_range must be a tuple as (min, max)"
        assert isinstance(new_range, tuple) and len(new_range) == 2, \
            "feature_range must be a tuple as (min, max)"

        min_start_value = starting_range[0]
        max_start_value = starting_range[1]
        min_new_value = new_range[0]
        max_new_value = new_range[1]

        value_std = (value - min_start_value) / (max_start_value - min_start_value)
        scaled_value = value_std * (max_new_value - min_new_value) + min_new_value

        return scaled_value


########################################################################################################################


class SingleStepVPPEnv(VPPEnv):
    """
    Gym environment for the single step version VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render_modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None,
                 logger=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        super(SingleStepVPPEnv, self).__init__(predictions, c_grid, shift, noise_std_dev, savepath, logger)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=0, high=np.inf, shape=(self.n * 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)
        self.history = {'c_virt': [], 'energy_bought': [], 'energy_sold': [], 'diesel_power': [],
                        'input_storage': [], 'output_storage': [], 'storage_capacity': []}

    def _get_observations(self) -> np.array:
        """
        Return predicted pv and load values as a single array.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.p_ren_pv_pred / np.max(self.p_ren_pv_pred),
                                       self.tot_cons_pred / np.max(self.tot_cons_pred)),
                                      axis=0)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.p_ren_pv_pred = None
        self.p_ren_pv_real = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None
        self.history = {'c_virt': [], 'energy_bought': [], 'energy_sold': [], 'diesel_power': [],
                        'input_storage': [], 'output_storage': [], 'storage_capacity': []}

    def _solve(self, c_virt: np.array) -> Tuple[List[gurobipy.Model], bool]:
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (num_timesteps, ); the virtual costs multiplied to output storage variable.
        :return: list of gurobipy.Model, bool; a list with the solved optimization model and True if the problem is
                                               feasible, False otherwise.
        """

        # Check variables initialization
        assert self.mr is not None, "Instance index must be initialized"
        assert self.c_grid is not None, "c_grid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.p_ren_pv_real is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        cap, p_diesel, p_storage_in, p_storage_out, p_grid_in, p_grid_out, tilde_cons = \
            [[None] * self.n for _ in range(7)]

        # Initialize the storage capacitance
        cap_x = self.in_cap

        # Save all the optimization models in a list
        models = []

        # Loop for each timestep
        for i in range(self.n):
            # Create a Gurobi model
            mod = Model()

            # Build variables and define bounds
            p_diesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel_" + str(i))
            p_storage_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in_" + str(i))
            p_storage_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out_" + str(i))
            p_grid_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in_" + str(i))
            p_grid_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out_" + str(i))
            cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))

            # Shifts from Demand Side Energy Management System
            tilde_cons[i] = (self.shift[i] + self.tot_cons_real[i])

            # Power balance constraint
            mod.addConstr((self.p_ren_pv_real[i] + p_storage_out[i] + p_grid_out[i] + p_diesel[i] -
                           p_storage_in[i] - p_grid_in[i] == tilde_cons[i]), "Power_balance")

            # Storage capacity constraint
            mod.addConstr(cap[i] == cap_x + p_storage_in[i] - p_storage_out[i])
            mod.addConstr(cap[i] <= self.cap_max)

            mod.addConstr(p_storage_in[i] <= self.cap_max - cap_x)
            mod.addConstr(p_storage_out[i] <= cap_x)

            mod.addConstr(p_storage_in[i] <= 200)
            mod.addConstr(p_storage_out[i] <= 200)

            # Diesel and grid bounds
            mod.addConstr(p_diesel[i] <= self.p_diesel_max)
            mod.addConstr(p_grid_in[i] <= 600)

            # Objective function
            obf = (self.c_grid[i] * p_grid_out[i] + self.c_diesel * p_diesel[i] +
                   c_virt[i] * p_storage_in[i] - self.c_grid[i] * p_grid_in[i])
            mod.setObjective(obf)

            feasible = VPPEnv.solve(mod)

            # If one of the timestep is not feasible, get out of the loop
            if not feasible:
                break

            models.append(mod)
            old_cap = cap_x
            # Update the storage capacity
            cap_x = cap[i].X
            for k, v in (
                    ('c_virt', c_virt[i]), ('energy_bought', p_grid_out[i].X), ('energy_sold', p_grid_in[i].X),
                    ('diesel_power', p_diesel[i].X),
                    ('input_storage', p_storage_in[i].X), ('output_storage', p_storage_out[i].X),
                    ('storage_capacity', old_cap)):
                self.history[k].append(v)

        return models, feasible

    def _compute_real_cost(self, models: List[gurobipy.Model]) -> Union[float, int]:
        """
        Given a list of optimization models, one for each timestep, the method returns the real cost value.
        :param models: list of gurobipy.Model; a list with an optimization model for each timestep.
        :return: float; the real cost of the given solution.
        """

        # Check that the number of optimization models is equal to the number of timestep
        assert len(models) == self.n

        cost = 0
        all_cost = []

        # Compute the total cost considering all the timesteps
        for timestep, model in enumerate(models):
            optimal_p_grid_out = model.getVarByName('p_grid_out_' + str(timestep)).X
            optimal_p_diesel = model.getVarByName('p_diesel_' + str(timestep)).X
            optimal_p_grid_in = model.getVarByName('p_grid_in_' + str(timestep)).X

            cost += (self.c_grid[timestep] * optimal_p_grid_out + self.c_diesel * optimal_p_diesel
                     - self.c_grid[timestep] * optimal_p_grid_in)
            all_cost.append(cost)

        return cost

    def step(self, action: np.array) -> Tuple[np.array, int | float, bool, bool, dict]:
        """
        This is a step performed in the environment: the virtual costs are set by the agent and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: numpy.array, float, boolean, dict; the observations, the reward, a boolean that is True if the episode
                                                    is ended, additional information.
        """
        terminated, truncated = False, False
        # Solve the optimization model with the virtual costs
        models, feasible = self._solve(action)

        if not feasible:
            reward = self.MIN_REWARD
            truncated = True
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)
            terminated = True

        observations = self._get_observations()

        return observations, reward, terminated, truncated, {'feasible': feasible, 'true cost': -reward}


########################################################################################################################


class MarkovianVPPEnv(VPPEnv):
    """
    Gym environment for the Markovian version of the VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render_modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None,
                 logger=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        super(MarkovianVPPEnv, self).__init__(predictions, c_grid, shift, noise_std_dev, savepath, logger)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n * 3 + 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.history = {'c_virt': [], 'energy_bought': [], 'energy_sold': [], 'diesel_power': [],
                        'input_storage': [], 'output_storage': [], 'storage_capacity': []}

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        super()._create_instance_variables()

        # Set the timestep
        self.timestep = 0

        # Set the cumulative cost
        self.cumulative_cost = 0

    def _get_observations(self) -> np.array:
        """
        Return predicted pv and load values, one-hot encoding of the timestep and the storage.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.p_ren_pv_pred.copy(), self.tot_cons_pred.copy()), axis=0)
        one_hot_timestep = np.zeros(shape=(self.n,))
        one_hot_timestep[int(self.timestep)] = 1
        observations = np.concatenate((observations, one_hot_timestep), axis=0)
        observations = np.append(observations, self.storage)
        observations = np.append(observations, self.cumulative_cost)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.p_ren_pv_pred = None
        self.p_ren_pv_real = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None
        self.storage = self.in_cap
        self.timestep = 0
        self.cumulative_cost = 0
        self.history = {'c_virt': [], 'energy_bought': [], 'energy_sold': [], 'diesel_power': [],
                        'input_storage': [], 'output_storage': [], 'storage_capacity': []}

    def _solve(self, c_virt: np.array) -> tuple[Model, bool]:
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (num_timesteps, ); the virtual costs multiplied to output storage variable.
        :return: list of gurobipy.Model, bool; a list with the solved optimization model and True if the model is
                                               feasible, False otherwise.
        """

        # Check variables initialization
        assert self.mr is not None, "Instance index must be initialized"
        assert self.c_grid is not None, "c_grid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.p_ren_pv_real is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"
        assert self.storage is not None, "Storage variable must be initialized"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        c_virt = np.squeeze(c_virt)

        # Create an optimization model
        mod = Model()

        # build variables and define bounds
        p_diesel = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel")
        p_storage_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in")
        p_storage_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out")
        p_grid_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in")
        p_grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out")
        cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap")

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Power balance constraint
        mod.addConstr((self.p_ren_pv_real[self.timestep] + p_storage_out + p_grid_out + p_diesel -
                       p_storage_in - p_grid_in == tilde_cons), "Power_balance")

        # Storage cap
        mod.addConstr(cap == self.storage + p_storage_in - p_storage_out)
        mod.addConstr(cap <= self.cap_max)

        mod.addConstr(p_storage_in <= self.cap_max - self.storage)
        mod.addConstr(p_storage_out <= self.storage)

        mod.addConstr(p_storage_in <= 200)
        mod.addConstr(p_storage_out <= 200)

        # Diesel and grid bounds
        mod.addConstr(p_diesel <= self.p_diesel_max)
        mod.addConstr(p_grid_in <= 600)

        # Objective function
        obf = (self.c_grid[self.timestep] * p_grid_out + self.c_diesel * p_diesel +
               c_virt * p_storage_in - self.c_grid[self.timestep] * p_grid_in)
        mod.setObjective(obf)

        feasible = VPPEnv.solve(mod)

        old_cap = self.storage
        # Update the storage capacitance
        if feasible:
            self.storage = cap.X
        # update history
        for k, v in (
                ('c_virt', c_virt), ('energy_bought', p_grid_out.X), ('energy_sold', p_grid_in.X),
                ('diesel_power', p_diesel.X),
                ('input_storage', p_storage_in.X), ('output_storage', p_storage_out.X), ('storage_capacity', old_cap)):
            self.history[k].append(v)
        return mod, feasible

    def _compute_real_cost(self, model: gurobipy.Model) -> Union[float, int]:
        """
        Given a list of models, one for each timestep, the method returns the real cost value.
        :param model: gurobipy.Model; optimization model for the current timestep.
        :return: float; the real cost of the current timestep.
        """

        optimal_p_grid_out = model.getVarByName('p_grid_out').X
        optimal_p_diesel = model.getVarByName('p_diesel').X
        optimal_p_grid_in = model.getVarByName('p_grid_in').X
        optimal_p_storage_in = model.getVarByName('p_storage_in').X
        optimal_p_storage_out = model.getVarByName('p_storage_out').X
        optimal_cap = model.getVarByName('cap').X

        cost = (self.c_grid[self.timestep] * optimal_p_grid_out + self.c_diesel * optimal_p_diesel
                - self.c_grid[self.timestep] * optimal_p_grid_in)

        return cost

    def step(self, action: np.array) -> Tuple[np.array, Union[float, int], bool, bool, dict]:
        """
        This is a step performed in the environment: the virtual costs are set by the agent and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: numpy.array, float, boolean, dict; the observations, the reward, a boolean that is True if the episode
                                                    is ended, additional information.
        """
        # Solve the optimization model with the virtual costs
        models, feasible = self._solve(action)

        if not feasible:
            reward = self.MIN_REWARD
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)

        # Update the cumulative cost
        self.cumulative_cost -= reward

        observations = self._get_observations()

        # Update the timestep
        self.timestep += 1
        assert self.timestep <= self.n, f"Timestep cannot be greater than {self.n}"
        terminated = (self.timestep == self.n)
        truncated = not feasible
        # done = False
        # if self.timestep == self.n or not feasible:
        #     done = True
        # elif self.timestep < self.n:
        #     done = False
        # else:
        #     raise Exception(f"Timestep cannot be greater than {self.n}")
        if (terminated or truncated) and self._do_log:
            self.log()
        return observations, reward, terminated, truncated, {'feasible': feasible, 'true cost': self.cumulative_cost}

    def log(self, ):
        """
        Logs training info using wandb.
        :return:
        """
        if self.logger is not None:
            means = dict()
            for ax, (k, hist) in enumerate(self.history.items()):
                means[f'avg_{k}'] = np.mean(hist)
            self.logger.log(prefix='env_final_eval', **means)


########################################################################################################################

def make_env(method,
             predictions,
             shift,
             c_grid,
             noise_std_dev,
             logger=None):
    # Set episode length and discount factor for single-step and MDP version
    if 'sequential' in method:
        max_episode_length = TIMESTEP_IN_A_DAY
        discount = 0.99
    elif 'all-at-once' in method:
        max_episode_length = 1
        discount = 0
    else:
        raise Exception("Method name must contain 'mdp' or 'single-step'")

    if method == 'unify-sequential':
        # Create the environment
        env = MarkovianVPPEnv(predictions=predictions,
                              shift=shift,
                              c_grid=c_grid,
                              noise_std_dev=noise_std_dev,
                              savepath=None,
                              logger=logger)

        # Garage wrapping of a gym environment
        # env = GymEnv(env, max_episode_length=max_episode_length)
        # env = NormalizeObservation(env)
    elif method == 'unify-all-at-once':
        # Create the environment
        env = SingleStepVPPEnv(predictions=predictions,
                               shift=shift,
                               c_grid=c_grid,
                               noise_std_dev=noise_std_dev,
                               savepath=None,
                               logger=logger)

        # Garage wrapping of a gym environment
        # env = GymEnv(env, max_episode_length=max_episode_length)
    else:
        raise NotImplementedError()

    return env, discount, max_episode_length


