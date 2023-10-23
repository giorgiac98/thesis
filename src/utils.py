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
    SafeModule, TanhDelta, AdditiveGaussianWrapper, TanhModule, SafeSequential
from torchrl.objectives import ClipPPOLoss, SACLoss, SoftUpdate, TD3Loss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.wandb import WandbLogger
from tqdm import tqdm
import wandb
from envs.vpp_envs import make_env
from envs.generate_instances import MinSetCoverEnv
import matplotlib.pyplot as plt

EVALUATION_MODES = {'mean': ExplorationType.MEAN, 'random': ExplorationType.RANDOM}


def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MyWandbLogger(WandbLogger):

    def __init__(
            self,
            exp_name: str = None,
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


def define_metrics(cfg, logger):
    problem_metrics = {
        'msc': [],
        'ems': ['eval/episode_reward', 'eval/episode_reward_values', 'eval/optimality', 'train/mean_episode_reward',
                'train/optimality'],
    }
    am = {
        'ppo': ['train/loss_objective', 'train/loss_critics', 'train/loss_entropy',
                'debug/ESS', 'debug/entropy'],
        'sac': ['train/loss_actor', 'train/loss_qvalue', 'train/loss_alpha',
                'debug/alpha', 'debug/entropy'],
        'td3': ['train/loss_critic'],
    }
    algo_metrics = {k: [(x, 'train/updates') for x in v] for k, v in am.items()}
    algo_metrics['td3'].append(('train/loss_actor', 'train/actor_updates'))

    for metric in problem_metrics[cfg.data.problem]:
        logger.experiment.define_metric(metric, summary="max", step_metric='train/iteration')

    logger.experiment.define_metric("train/collected_frames", step_metric='train/iteration')
    logger.experiment.define_metric("train/episodes", step_metric='train/iteration')

    logger.experiment.define_metric("eval/action", step_metric='train/iteration')
    logger.experiment.define_metric("debug/action", step_metric='train/iteration')
    logger.experiment.define_metric("debug/actor_lr", step_metric='train/iteration')
    logger.experiment.define_metric("debug/critic_lr", step_metric='train/iteration')
    logger.experiment.define_metric("debug/loc", step_metric='train/iteration')
    logger.experiment.define_metric("debug/scale", step_metric='train/iteration')

    logger.experiment.define_metric("train/grad_norm", step_metric='train/updates')

    for metric, step in algo_metrics[cfg.model.policy]:
        logger.experiment.define_metric(metric, step_metric=step)


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
                                           default_interaction_type=InteractionType.RANDOM,
                                           )

        module = QValueModule(obs_spec=input_shape,
                              act_spec=n_action,
                              device=device,
                              net_spec=value_net_spec)
        qvalue = ValueOperator(module=module, in_keys=['observation', 'action'])
        loss_module = SACLoss(policy_module, qvalue, **policy_spec)
        loss_module.make_value_estimator(gamma=policy_spec.gamma)
        # Define Target Network Updater
        target_net_updater = SoftUpdate(loss_module, tau=other_spec.target_update_polyak)
        other = {'qvalue': qvalue, 'target_net_updater': target_net_updater}
    elif policy == 'td3':
        dist_kwargs = {
            "low": -20,  # env_action_spec.space.minimum,
            "high": 20,  # env_action_spec.space.maximum,
        }
        net = MLP(in_features=input_shape,
                  out_features=n_action,
                  device=device,
                  depth=actor_net_spec.depth,
                  num_cells=actor_net_spec.num_cells,
                  activation_class=get_activation(actor_net_spec.activation))
        module = SafeModule(net, in_keys=["observation"], out_keys=["param"])
        policy_module = SafeSequential(
            module,
            TanhModule(
                in_keys=["param"],
                out_keys=["action"],
                **dist_kwargs,
            ),
        )
        # OLD
        # policy_module = ProbabilisticActor(
        #     spec=env_action_spec,
        #     in_keys=["param"],
        #     module=module,
        #     distribution_class=TanhDelta,
        #     distribution_kwargs=dist_kwargs,
        #     default_interaction_type=InteractionType.RANDOM,
        #     return_log_prob=False)

        q_value_net = QValueModule(obs_spec=input_shape,
                                   act_spec=1,
                                   device=device,
                                   net_spec=value_net_spec)
        qvalue = ValueOperator(module=q_value_net, in_keys=['observation', 'action'])
        actor_model_explore = AdditiveGaussianWrapper(
            policy_module,
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=0.1,
            safe=False,
        ).to(device)
        loss_module = TD3Loss(
            actor_network=policy_module,
            qvalue_network=qvalue,
            bounds=(-20., 20.),
            **policy_spec,
        )
        loss_module.make_value_estimator(gamma=policy_spec.gamma)

        # Define Target Network Updater
        target_net_updater = SoftUpdate(loss_module, tau=other_spec.target_update_polyak)
        other = {'qvalue': qvalue, 'target_net_updater': target_net_updater,
                 'actor_model_explore': actor_model_explore}
    else:
        raise NotImplementedError(f"Policy {policy} not implemented")
    return policy_module, loss_module, other


def make_optimizer(cfg, loss_module):
    if cfg.model.policy != 'td3':
        lambdas = [lambda _: cfg.schedule_factor] * 2
        if cfg.model.policy == 'ppo':
            splitted = [
                {'params': [p for k, p in loss_module.named_parameters() if 'actor' in k], 'lr': cfg.actor_lr},
                {'params': [p for k, p in loss_module.named_parameters() if 'critic' in k], 'lr': cfg.critic_lr},
            ]
        else:
            splitted = [
                {'params': list(loss_module.actor_network_params.flatten_keys().values()), 'lr': cfg.actor_lr},
                {'params': list(loss_module.qvalue_network_params.flatten_keys().values()), 'lr': cfg.critic_lr},
                {'params': [loss_module.log_alpha], 'lr': 3.0e-3}
            ]
            lambdas.append(lambda _: 1.)
        optim = torch.optim.Adam(splitted)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lambdas) if cfg.schedule_lr else None
    else:
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        optimizer_actor = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
        optimizer_critic = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
        optim = (optimizer_actor, optimizer_critic)
        scheduler = [torch.optim.lr_scheduler.MultiplicativeLR(op, lr_lambda=lambda _: cfg.schedule_factor)
                     if cfg.schedule_lr else None
                     for op in optim]
    return optim, scheduler


def training_loop(cfg, policy_module, loss_module, other_modules, optim, collector, replay_buffer, device,
                  test_env, logger=None, scheduler=None):
    logs = defaultdict(list)
    logs['episodes'] = 0
    logs['collected_frames'] = 0
    logs['updates'] = 0
    logs['actor_updates'] = 0
    logs['best_episode_reward'] = -np.inf
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
                    compute_loss(cfg, device, logs, logger, loss_module, optim, batch)
            if scheduler is not None:
                scheduler.step()

            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost, i,
                                  do_log_step=i % cfg.eval_interval != 0)
            if i % cfg.eval_interval == 0:
                eval_str = evaluate_policy(cfg, logger, logs, policy_module, test_env)
            pbar.set_description(", ".join([eval_str, *logs_str]))

    else:
        target_net_updater = other_modules['target_net_updater']
        for i, tensordict_data in enumerate(collector):
            logs['collected_frames'] += tensordict_data.numel()

            if cfg.model.policy == 'td3':
                other_modules['actor_model_explore'].step(tensordict_data.numel())
            collector.update_policy_weights_()
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            if logs['collected_frames'] >= cfg.model.other_spec.init_random_frames:
                for _ in range(cfg.update_rounds):
                    # sample from replay buffer
                    sampled_tensordict = replay_buffer.sample().clone()
                    if cfg.model.policy == 'sac':
                        compute_loss(cfg, device, logs, logger, loss_module, optim, sampled_tensordict)
                        target_net_updater.step()
                    else:  # TD3
                        sampled_tensordict = compute_td3_loss(cfg, logger, logs, loss_module, optim, sampled_tensordict,
                                                              target_net_updater)
                    # update priority
                    if cfg.prb:
                        replay_buffer.update_tensordict_priority(sampled_tensordict)
                if cfg.model.policy == 'sac':
                    if scheduler is not None:
                        scheduler.step()
                else:
                    for s in scheduler:
                        if s is not None:
                            s.step()
            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost, i,
                                  do_log_step=i % cfg.eval_interval != 0)
            if i % cfg.eval_interval == 0:
                eval_str = evaluate_policy(cfg, logger, logs, policy_module, test_env)
            pbar.set_description(", ".join([eval_str, *logs_str]))

    collector.shutdown()
    pbar.close()


def compute_td3_loss(cfg, logger, logs, loss_module, optim, sampled_tensordict, target_net_updater):
    optimizer_actor, optimizer_critic = optim
    logs['updates'] += 1
    update_actor = logs['updates'] % cfg.model.other_spec.policy_delay_update == 0
    # Compute loss
    q_loss, other_q = loss_module.value_loss(sampled_tensordict)
    # Update critic
    optimizer_critic.zero_grad()
    q_loss.backward()
    optimizer_critic.step()
    # TODO this is a bug fix for the TD3 implementation, tell torchrl
    sampled_tensordict = sampled_tensordict.set('td_error', other_q['td_error'].detach().max(0)[0])
    # Update actor
    if update_actor:
        logs['actor_updates'] += 1
        actor_loss, other_a = loss_module.actor_loss(sampled_tensordict)
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # Update target params
        target_net_updater.step()
        if logger is not None:
            logger.log(do_log_step=False, prefix="train", loss_actor=actor_loss.item(),
                       actor_updates=logs['actor_updates'])
            logger.log(do_log_step=False, prefix="debug", **other_a)
    if logger is not None:
        logger.log(do_log_step=False, prefix="train", loss_critic=q_loss.item(),
                   updates=logs['updates'])
        logger.log(prefix="debug", **other_q)
    return sampled_tensordict


def evaluate_policy(cfg, logger, logs, policy_module, test_env):
    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollouts = torch.stack([test_env.rollout(cfg.eval_rollout_steps, policy_module)
                                     for _ in range(cfg.eval_rollouts)])
        episode_reward = eval_rollouts[('next', 'episode_reward')][:, -1].mean().item()
        optimality = -test_env.optimal_cost / episode_reward
        if logger is not None:
            logger.log(prefix="eval", episode_reward=episode_reward, optimality=optimality,
                       episode_reward_values=wandb.Histogram(
                           np_histogram=np.histogram(eval_rollouts[('next', 'episode_reward')][:, -1])),
                       action=wandb.Histogram(np_histogram=np.histogram(eval_rollouts['action'].mean(dim=0))))
        logs["eval reward"].append(eval_rollouts["next", "reward"].mean().item())
        logs["eval reward (sum)"].append(episode_reward)
        logs["eval step_count"].append(eval_rollouts["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(best: {logs['best_episode_reward']: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )

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


def train_logs(logger, logs, optim, pbar, tensordict_data, optimal_cost, it, do_log_step=True):
    dones = tensordict_data[('next', 'done')]
    mean_episode_reward = tensordict_data[('next', 'episode_reward')][dones].mean().item()
    optimality = -optimal_cost / mean_episode_reward
    logs['episodes'] += dones.sum().item()
    if isinstance(optim, tuple):
        actor_lr = optim[0].param_groups[0]["lr"]
        critic_lr = optim[1].param_groups[0]["lr"]
    else:
        actor_lr = optim.param_groups[0]["lr"]
        critic_lr = optim.param_groups[1]["lr"]
    if logger is not None:
        logger.log(do_log_step=False, prefix="train", mean_episode_reward=mean_episode_reward,
                   optimality=optimality, episodes=logs['episodes'], collected_frames=logs['collected_frames'],
                   iteration=it)
        t_keys = tensordict_data.keys()
        if 'loc' in t_keys and 'scale' in t_keys:
            p = {'loc': wandb.Histogram(np_histogram=np.histogram(tensordict_data['loc'].reshape(-1, 1))),
                 'scale': wandb.Histogram(np_histogram=np.histogram(tensordict_data['scale'].reshape(-1, 1)))}
        else:
            p = {'param': wandb.Histogram(np_histogram=np.histogram(tensordict_data['param'].reshape(-1, 1)))}
        logger.log(do_log_step=do_log_step, prefix="debug", actor_lr=actor_lr, critic_lr=critic_lr, **p,
                   action=wandb.Histogram(np_histogram=np.histogram(tensordict_data['action'].reshape(-1, 1))))
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["act_lr"].append(actor_lr)
    logs["cri_lr"].append(critic_lr)
    lr_str = f"lr policy: {actor_lr: 4.4f}"
    return cum_reward_str, lr_str, stepcount_str


def compute_loss(cfg, device, logs, logger, loss_module, optim, subdata):
    loss_vals = loss_module(subdata.to(device))
    losses = {key: loss for key, loss in loss_vals.items() if key.startswith("loss_")}
    loss_value = sum(losses.values())
    # Optimization: backward, grad clipping and optim step
    loss_value.backward()
    clipped_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
    optim.step()
    optim.zero_grad()

    logs['updates'] += 1
    if logger is not None:
        logger.log(do_log_step=False, prefix="train", **losses, grad_norm=clipped_norm, updates=logs['updates'])
        logger.log(prefix="debug", **{key: loss for key, loss in loss_vals.items() if not key.startswith("loss_")})


def final_evaluation(cfg, logger, policy_module, test_env, test_dir=None):
    name = get_model_name(cfg)
    dir_name = get_dir_name(cfg, logger) if test_dir is None else test_dir
    policy_module.load_state_dict(torch.load(f'{dir_name}/{name}'))
    policy_module.eval()  # might not be necessary

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        eval_rollout = test_env.rollout(100, policy_module)
        episode_reward = eval_rollout[('next', 'episode_reward')][-1].mean().item()
        optimality = -test_env.optimal_cost / episode_reward
        test_solution = test_env.history

    optimal_solution_df = test_env.optimal_solution
    axes = optimal_solution_df.plot(subplots=True, fontsize=12, figsize=(10, 7))
    plt.xlabel('Timestamp', fontsize=14)

    for axis in axes:
        axis.legend(loc=2, prop={'size': 12})
    plt.plot()
    if logger is not None:
        optimal_solution_df['Timestep'] = list(range(len(optimal_solution_df)))

        d = {hs: wandb.plot.line_series(
            xs=optimal_solution_df['Timestep'].to_numpy(),
            ys=[optimal_solution_df[df], test_solution[hs]],
            keys=["optimal action", "learned action"],
            title=df,
            xname="timesteps")
            for hs, df in (('energy_bought', 'Energy bought'), ('energy_sold', 'Energy sold'),
                           ('diesel_power', 'Diesel power consumption'), ('storage_capacity', 'Storage capacity'),
                           ('input_storage', 'Input to storage'), ('output_storage', 'Output from storage'))}
        action_table = wandb.Table(data=[[x, float(y)] for (x, y) in zip(optimal_solution_df['Timestep'],
                                                                         test_solution['c_virt'])],
                                   columns=["timestep", "action"])
        d["actions"] = wandb.plot.line(action_table, "timestep", "action",
                                       title="C_virt Actions")
        logger.log(prefix='final_eval', episode_reward=episode_reward, optimality=optimality, **d,
                   opt_chart=axes[0].get_figure(), opt_chart_data=wandb.Table(dataframe=optimal_solution_df))


def get_dir_name(cfg, logger=None):
    return logger.experiment.dir if logger is not None else f'outputs/{cfg.data.problem}/{cfg.model.policy}'


def get_model_name(cfg):
    return f"best_{cfg.model.policy}_policy.pt"
