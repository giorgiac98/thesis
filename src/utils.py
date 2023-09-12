from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule, InteractionType
from torch import nn
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, StepCounter, check_env_specs, set_exploration_type, \
    ExplorationType, RewardSum, RewardScaling, default_info_dict_reader
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, IndependentNormal, ValueOperator, NormalParamWrapper, \
    SafeModule
from torchrl.objectives import ClipPPOLoss, SACLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl.record.loggers.wandb import WandbLogger
from tqdm import tqdm
import wandb
from envs.vpp_envs import make_env
from envs.generate_instances import MinSetCoverEnv
import matplotlib.pyplot as plt


class MyWandbLogger(WandbLogger):

    def __init__(
            self,
            exp_name: str,
            offline: bool = False,
            save_dir: str = None,
            id: str = None,
            project: str = None,
            **kwargs,
    ):
        super().__init__(exp_name, offline, save_dir, id, project, **kwargs)
        self._log_dict = dict()

    def log(self, do_log_step: bool = True, prefix: str = '', **kwargs) -> None:
        """Logs all kwargs to WandB.

            Args:
                do_log_step: (Optional) If False, caches data and waits next _log() call to perform actual logging.
                  Defaults to True.
                prefix: (Optional) A str that is prefixed to all keys to be logged.
                  Defaults to "" (empty string, no prefix).
        """
        assert self.experiment, 'must init logger by passing wandb parameters at construction time'
        log_dict = {f'{prefix}{"/" if prefix else ""}{k}': v for k, v in kwargs.items()}
        self._log_dict.update(log_dict)
        if do_log_step:
            self.experiment.log(self._log_dict)
            self._log_dict = {}


class QValueModule(nn.Module):
    """
    Q-Network module for continuous action spaces
    Takes as input the observation and action and outputs the Q-value
    """

    def __init__(self, obs_spec, act_spec, device, net_spec):
        super().__init__()
        self.mlp = MLP(in_features=obs_spec + act_spec,
                       out_features=1,
                       device=device,
                       depth=net_spec.depth,
                       num_cells=net_spec.num_cells,
                       activation_class=get_activation(net_spec.activation))

    def forward(self, obs, act):
        return self.mlp(torch.cat([obs, act], -1))


def get_activation(act: str):
    if act == "relu":
        return nn.ReLU
    elif act == "tanh":
        return nn.Tanh
    elif act == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Activation {act} not implemented")


def env_maker(problem: str,
              params: dict,
              device: torch.device):
    assert problem in ['msc', 'ems'], f"Environment not implemented for {problem} problem"
    if problem == 'ems':
        predictions = pd.read_csv(params['predictions_filepath'])
        shift = np.load(params['shifts_filepath'])
        c_grid = np.load(params['prices_filepath'])
        instance = params['instance']
        info_keys = []

        def create_env(logger=None):
            base_env, _, _ = make_env(f'unify-{params["method"]}',
                                      predictions.iloc[[instance]],
                                      shift,
                                      c_grid,
                                      params['noise_std_dev'],
                                      logger)
            return base_env

        env_creator = create_env
    elif problem == 'msc':
        info_keys = []
        env_creator = lambda logger: MinSetCoverEnv(num_prods=params['num_prods'],
                                                    num_sets=params['num_sets'],
                                                    instances_filepath=params['data_path'],
                                                    seed=params['seed'])
    else:
        raise NotImplementedError

    def make_init_env(logger=None, state_dict=None):
        base_env = env_creator(logger)
        torchrl_env = GymWrapper(base_env, device=device)
        torchrl_env.set_info_dict_reader(default_info_dict_reader(info_keys))
        e = TransformedEnv(
            torchrl_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                StepCounter(),
                RewardSum(),
                RewardScaling(0, 0.05)
            ),
        )
        if state_dict is not None:
            e.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0)
            e.transform[0].load_state_dict(state_dict)
        else:
            e.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
        check_env_specs(e)
        return e

    return make_init_env


def prepare_networks_and_policy(policy, policy_spec, other_spec, actor_net_spec, value_net_spec, device,
                                env_action_spec,
                                input_shape):
    n_action = env_action_spec.shape[-1]
    if policy == 'ppo':
        actor_mlp = MLP(in_features=input_shape,
                        out_features=2 * n_action,
                        device=device,
                        depth=actor_net_spec.depth,
                        num_cells=actor_net_spec.num_cells,
                        activation_class=get_activation(actor_net_spec.activation))
        actor_net = nn.Sequential(
            actor_mlp,
            NormalParamExtractor(),
        )
        torch.nn.init.uniform_(actor_net[0][-1].weight, -1e-3, 1e-3)
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=env_action_spec,
            in_keys=["loc", "scale"],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
        value_net = MLP(in_features=input_shape,
                        out_features=1,
                        device=device,
                        depth=value_net_spec.depth,
                        num_cells=value_net_spec.num_cells,
                        activation_class=get_activation(value_net_spec.activation))
        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
        advantage_module = GAE(value_network=value_module, **other_spec.gae_spec)
        loss_module = ClipPPOLoss(
            actor=policy_module,
            critic=value_module,
            **policy_spec,
            entropy_bonus=bool(policy_spec['entropy_coef']),
            value_target_key=advantage_module.value_target_key,
        )
        other = {'value': value_module, 'advantage': advantage_module}
    elif policy == 'sac':
        net = NormalParamWrapper(MLP(in_features=input_shape,
                                     out_features=2 * n_action,
                                     device=device,
                                     depth=actor_net_spec.depth,
                                     num_cells=actor_net_spec.num_cells,
                                     activation_class=get_activation(actor_net_spec.activation)))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        dist_kwargs = {
            "min": -20,  # env_action_spec.space.minimum,
            "max": 20,  # env_action_spec.space.maximum,
            "tanh_loc": False,
        }
        # FIXME: è probabile che min e max lasciati di default a -1,1 siano la causa del problema,
        # ma settando -inf, inf non funziona perché restituisce nan
        policy_module = ProbabilisticActor(module=module,
                                           in_keys=["loc", "scale"],
                                           spec=env_action_spec,
                                           distribution_class=TanhNormal,
                                           distribution_kwargs=dist_kwargs,
                                           # default_interaction_type=InteractionType.RANDOM
                                           )

        module = QValueModule(obs_spec=input_shape,
                              act_spec=n_action,
                              device=device,
                              net_spec=value_net_spec)
        qvalue = ValueOperator(module=module, in_keys=['observation', 'action'])
        loss_module = SACLoss(policy_module, qvalue, **policy_spec)
        loss_module.make_value_estimator(gamma=policy_spec.gamma)
        # Define Target Network Updater
        target_net_updater = SoftUpdate(
            loss_module, eps=other_spec.target_update_polyak
        )
        other = {'qvalue': qvalue, 'target_net_updater': target_net_updater}
    else:
        raise NotImplementedError(f"Policy {policy} not implemented")
    return policy_module, loss_module, other


def training_loop(cfg, policy_module, loss_module, other_modules, optim, collector, replay_buffer, device,
                  test_env, logger=None, scheduler=None):
    if logger is not None:
        logger.experiment.define_metric("eval/episode_reward", summary="max")
    logs = defaultdict(list)
    logs['episodes'] = 0
    logs['collected_frames'] = 0
    logs['updates'] = 0
    logs['best_episode_reward'] = None
    optimal_cost = test_env.optimal_cost
    pbar = tqdm(total=collector.total_frames)
    eval_str = ""
    if cfg.model.policy == 'ppo':
        advantage_module = other_modules['advantage']
        for i, tensordict_data in enumerate(collector):
            logs['collected_frames'] += tensordict_data.numel()
            for _ in range(cfg.update_rounds):
                with torch.no_grad():
                    advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())
                for j, batch in enumerate(replay_buffer):
                    compute_loss(cfg, device, logger, loss_module, optim, batch)

            logs['updates'] += cfg.update_rounds
            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost)
            if i % cfg.eval_interval == 0:
                eval_str = evaluate_policy(cfg, logger, logs, policy_module, test_env)
            pbar.set_description(", ".join([eval_str, *logs_str]))
            if scheduler is not None:
                scheduler.step()

    elif cfg.model.policy == 'sac':
        target_net_updater = other_modules['target_net_updater']
        for i, tensordict_data in enumerate(collector):
            logs['collected_frames'] += tensordict_data.numel()
            collector.update_policy_weights_()
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            if logs['collected_frames'] >= cfg.model.other_spec.init_random_frames:
                for _ in range(cfg.update_rounds):
                    # sample from replay buffer
                    sampled_tensordict = replay_buffer.sample().clone()
                    compute_loss(cfg, device, logger, loss_module, optim, sampled_tensordict)
                    target_net_updater.step()
                logs['updates'] += cfg.update_rounds
                if scheduler is not None:
                    scheduler.step()
            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost)
            if i % cfg.eval_interval == 0:
                eval_str = evaluate_policy(cfg, logger, logs, policy_module, test_env)
            pbar.set_description(", ".join([eval_str, *logs_str]))

    collector.shutdown()
    pbar.close()


def evaluate_policy(cfg, logger, logs, policy_module, test_env):
    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollouts = torch.stack([test_env.rollout(cfg.eval_rollout_steps, policy_module)
                                     for _ in range(cfg.eval_rollouts)])
        episode_reward = eval_rollouts[('next', 'episode_reward')][:, -1].mean().item()
        optimality = -test_env.optimal_cost / episode_reward
        if logger is not None:
            logger.log(prefix="eval", episode_reward=episode_reward, optimality=optimality)
        logs["eval reward"].append(eval_rollouts["next", "reward"].mean().item())
        logs["eval reward (sum)"].append(episode_reward)
        logs["eval step_count"].append(eval_rollouts["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )
        if logs['best_episode_reward'] is None:
            logs['best_episode_reward'] = episode_reward

        if episode_reward > logs['best_episode_reward']:
            logs['best_episode_reward'] = episode_reward
            name = get_model_name(cfg)
            dir_name = get_dir_name(cfg, logger)
            torch.save(policy_module.state_dict(), f'{dir_name}/{name}')
            if logger is not None:
                wandb.save(f'{dir_name}/{name}')
        test_env.reset()  # reset the env after the eval rollout
        del eval_rollouts
    return eval_str


def train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost):
    dones = tensordict_data[('next', 'done')]
    mean_episode_reward = tensordict_data[('next', 'episode_reward')][dones].mean().item()
    optimality = -optimal_cost / mean_episode_reward
    logs['episodes'] += dones.sum().item()
    if logger is not None:
        logger.log(prefix="train", mean_episode_reward=mean_episode_reward, optimality=optimality,
                   lr=optim.param_groups[0]["lr"], episodes=logs['episodes'],
                   updates=logs['updates'], collected_frames=logs['collected_frames'])
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    return cum_reward_str, lr_str, stepcount_str


def compute_loss(cfg, device, logger, loss_module, optim, subdata):
    loss_vals = loss_module(subdata.to(device))
    loss_value = sum(loss for key, loss in loss_vals.items() if key.startswith("loss_"))
    # Optimization: backward, grad clipping and optim step
    loss_value.backward()
    clipped_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
    optim.step()
    optim.zero_grad()
    if logger is not None:
        logger.log(prefix="train", **loss_vals, grad_norm=clipped_norm)


def final_evaluation(cfg, logger, policy_module, test_env, test_dir=None):
    name = get_model_name(cfg)
    dir_name = get_dir_name(cfg, logger) if test_dir is None else test_dir
    policy_module.load_state_dict(torch.load(f'{dir_name}/{name}'))
    policy_module.eval()  # might not be necessary
    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        eval_rollout = test_env.rollout(100, policy_module)
        episode_reward = eval_rollout[('next', 'episode_reward')][-1].mean().item()
        optimality = -test_env.optimal_cost / episode_reward

        optimal_solution_df = test_env.optimal_solution
        test_solution = test_env.history
        axes = optimal_solution_df.plot(subplots=True, fontsize=12, figsize=(10, 7))
        plt.xlabel('Timestamp', fontsize=14)

        for axis in axes:
            axis.legend(loc=2, prop={'size': 12})
        plt.plot()
        if logger is not None:
            optimal_solution_df['Timestep'] = list(range(len(optimal_solution_df)))
            d = {hs: wandb.plot.line_series(
                xs=optimal_solution_df['Timestep'],
                ys=[test_solution[hs], optimal_solution_df[df]],
                keys=["learned action", "optimal action"],
                title=df,
                xname="timesteps")
                for hs, df in (('energy_bought', 'Energy bought'), ('energy_sold', 'Energy sold'),
                               ('diesel_power', 'Diesel power consumption'), ('storage_capacity', 'Storage capacity'),
                               ('input_storage', 'Input to storage'), ('output_storage', 'Output from storage'))}
            logger.log(prefix='final_eval', episode_reward=episode_reward, optimality=optimality,
                       opt_chart=plt, opt_chart_data=wandb.Table(dataframe=optimal_solution_df), **d)


def get_dir_name(cfg, logger=None):
    return logger.experiment.dir if logger is not None else f'outputs/{cfg.data.problem}/{cfg.model.policy}'


def get_model_name(cfg):
    return f"best_{cfg.model.policy}_policy.pt"
