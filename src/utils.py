import os

import numpy as np
import pandas as pd
import torch
from tensordict.nn import TensorDictModule, InteractionType, AddStateIndependentNormalScale
from torch import nn
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, StepCounter, check_env_specs, set_exploration_type, \
    ExplorationType, RewardSum, RewardScaling, default_info_dict_reader
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, IndependentNormal, ValueOperator, NormalParamWrapper, \
    SafeModule, AdditiveGaussianWrapper, TanhModule, SafeSequential
from torchrl.objectives import ClipPPOLoss, SACLoss, SoftUpdate, TD3Loss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.wandb import WandbLogger
from tqdm import tqdm
import wandb
from envs.vpp_envs import make_env
from envs.generate_instances import MinSetCoverEnv
import matplotlib.pyplot as plt


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
    default_metrics = ['eval/episode_reward', 'eval/episode_reward_values', 'eval/optimality', 'eval/optimality_values',
                       'best/optimality', 'best/episodes', 'best/updates',
                       'train/mean_episode_reward', 'train/optimality']
    problem_metrics = {
        'msc': ['eval/regret', 'eval/regret_std', 'eval/regret_values', 'debug/regret', 'debug/regret_values'],
        'ems': [],
    }
    am = {
        'ppo': ['train/loss_objective', 'train/loss_critics', 'train/loss_entropy',
                'debug/ESS', 'debug/entropy'],
        'sac': ['train/loss_actor', 'train/loss_qvalue', 'train/loss_alpha',
                'debug/alpha', 'debug/entropy'],
        'td3': ['train/loss_qvalue'],
    }
    algo_metrics = {k: [(x, 'train/updates') for x in v] for k, v in am.items()}
    algo_metrics['td3'].append(('train/loss_actor', 'train/actor_updates'))

    for metric in default_metrics:
        logger.experiment.define_metric(metric, summary="max", step_metric='train/iteration')

    for metric in problem_metrics:
        logger.experiment.define_metric(metric, summary="max", step_metric='train/iteration')

    logger.experiment.define_metric("train/collected_frames", step_metric='train/iteration')
    logger.experiment.define_metric("train/episodes", step_metric='train/iteration')

    logger.experiment.define_metric("eval/action", step_metric='train/iteration')
    logger.experiment.define_metric("eval/episode_reward_std", step_metric='train/iteration')
    logger.experiment.define_metric("eval/optimality_std", step_metric='train/iteration')
    logger.experiment.define_metric("debug/action", step_metric='train/iteration')
    logger.experiment.define_metric("debug/actor_lr", step_metric='train/iteration')
    logger.experiment.define_metric("debug/critic_lr", step_metric='train/iteration')
    logger.experiment.define_metric("debug/loc", step_metric='train/iteration')
    logger.experiment.define_metric("debug/scale", step_metric='train/iteration')
    logger.experiment.define_metric("debug/param", step_metric='train/iteration')

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
              device: torch.device,
              **kwargs):
    assert problem in ['msc', 'ems'], f"Environment not implemented for {problem} problem"
    info_keys = ['optimal_cost']
    spec = [UnboundedContinuousTensorSpec(shape=torch.Size([1]), dtype=torch.float64)]
    if problem == 'ems':
        predictions = pd.read_csv(params['predictions_filepath'])
        shift = np.load(params['shifts_filepath'])
        c_grid = np.load(params['prices_filepath'])
        instance = params['instance']

        def env_creator():
            base_env, _, _ = make_env(f'unify-{params["method"]}',
                                      predictions.iloc[[instance]],
                                      shift,
                                      c_grid,
                                      params['noise_std_dev'])
            return base_env

    elif problem == 'msc':
        # do not remove this line, in this way hydra resolves the path only once
        params['data_path'] = params['data_path']
        info_keys.append('demands')
        info_keys.append('mod_action')
        spec.append(UnboundedContinuousTensorSpec(shape=torch.Size([params['num_prods']]), dtype=torch.int64))
        spec.append(UnboundedContinuousTensorSpec(shape=torch.Size([params['num_prods']]), dtype=torch.float32))

        def env_creator():
            return MinSetCoverEnv(num_prods=params['num_prods'],
                                  num_sets=params['num_sets'],
                                  instances_filepath=params['data_path'],
                                  seed=kwargs['seed'])

    else:
        raise NotImplementedError

    def make_init_env(state_dict=None):
        base_env = env_creator()
        torchrl_env = GymWrapper(base_env, device=device)
        torchrl_env = torchrl_env.set_info_dict_reader(
            default_info_dict_reader(info_keys, spec=spec)
        )
        transforms = [ObservationNorm(in_keys=["observation"])] if params['obs_norm'] else []
        transforms += [StepCounter(), RewardSum(), RewardScaling(0, 0.05)]
        e = TransformedEnv(
            torchrl_env,
            Compose(*transforms),
        )
        if params['obs_norm']:
            if state_dict is not None:
                e.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0)
                e.transform[0].load_state_dict(state_dict)
            else:
                e.transform[0].init_stats(num_iter=500, reduce_dim=0, cat_dim=0)
        check_env_specs(e)
        return e

    return make_init_env


def prepare_networks_and_policy(policy, policy_spec, other_spec, actor_net_spec, value_net_spec,
                                device, problem_spec, env_action_spec, input_shape):
    n_action = env_action_spec.shape[-1]
    if policy == 'ppo':
        actor_net = MLP(in_features=input_shape,
                        out_features=n_action,  # outputs only loc
                        device=device,
                        depth=actor_net_spec.depth,
                        num_cells=actor_net_spec.num_cells,
                        activation_class=get_activation(other_spec.activation))
        # Initialize policy weights
        for layer in actor_net.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()
        # Add state-independent normal scale
        actor_net = torch.nn.Sequential(
            actor_net,
            AddStateIndependentNormalScale(n_action),
        )
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
                        activation_class=get_activation(other_spec.activation))
        # Initialize value weights
        for layer in value_net.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)
                layer.bias.data.zero_()
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
        torch.nn.init.uniform_(net.operator[-1].weight, -1e-3, 1e-3)
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])

        if problem_spec.use_tanh:
            min_ = problem_spec.low if 'low' in problem_spec else env_action_spec.space.minimum
            max_ = problem_spec.high if 'high' in problem_spec else env_action_spec.space.maximum
            dist_kwargs = {
                "min": min_,
                "max": max_,
                "tanh_loc": False,
            }
            dist_class = TanhNormal
        else:
            dist_class = IndependentNormal
            dist_kwargs = {}

        policy_module = ProbabilisticActor(module=module,
                                           in_keys=["loc", "scale"],
                                           spec=env_action_spec,
                                           distribution_class=dist_class,
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
        net = MLP(in_features=input_shape,
                  out_features=n_action,
                  device=device,
                  depth=actor_net_spec.depth,
                  num_cells=actor_net_spec.num_cells,
                  activation_class=get_activation(actor_net_spec.activation))
        module = SafeModule(net, in_keys=["observation"], out_keys=["param"])
        min_ = problem_spec.low if 'low' in problem_spec else env_action_spec.space.minimum
        max_ = problem_spec.high if 'high' in problem_spec else env_action_spec.space.maximum
        if problem_spec.use_tanh:
            dist_kwargs = {
                "low": min_,
                "high": max_,
            }
            policy_module = SafeSequential(
                module,
                TanhModule(
                    in_keys=["param"],
                    out_keys=["action"],
                    **dist_kwargs,
                ),
            )
        else:
            # TODO testare e vedere come performa (magari si può usare ReLu piuttosto che Identity)
            policy_module = SafeSequential(module,
                                           TensorDictModule(nn.Identity(), in_keys=["param"], out_keys=["action"]))
        q_value_net = QValueModule(obs_spec=input_shape,
                                   act_spec=n_action,
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
            bounds=(min_, max_),
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
                {'params': [loss_module.log_alpha], 'lr': cfg.model.other_spec.alpha_lr}
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
    logs = {"eval optimality": [], "eval reward (sum)": [], 'act_lr': [], 'cri_lr': [],
            'episodes': 0, 'collected_frames': 0, 'updates': 0, 'actor_updates': 0,
            'best_optimality': -1, 'best_episodes': 0, 'best_updates': 0}
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

            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, i,
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
            logs_str = train_logs(logger, logs, optim, pbar, tensordict_data, i,
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
    q_loss, q_metadata = loss_module.value_loss(sampled_tensordict)
    # Update critic
    optimizer_critic.zero_grad()
    q_loss.backward()
    optimizer_critic.step()
    # TODO this is a bug fix for the TD3 implementation, tell torchrl
    sampled_tensordict = sampled_tensordict.set('td_error', q_metadata['td_error'].detach().max(0)[0])
    # Update actor
    if update_actor:
        logs['actor_updates'] += 1
        actor_loss, a_metadata = loss_module.actor_loss(sampled_tensordict)
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # Update target params
        target_net_updater.step()
        if logger is not None:
            logger.log(do_log_step=False, prefix="train", loss_actor=actor_loss.item(),
                       actor_updates=logs['actor_updates'])
            logger.log(do_log_step=False, prefix="debug", **a_metadata)
    if logger is not None:
        logger.log(do_log_step=False, prefix="train", loss_critic=q_loss.item(),
                   updates=logs['updates'])
        logger.log(prefix="debug", **q_metadata)
    return sampled_tensordict


def train_logs(logger, logs, optim, pbar, tensordict_data, it, do_log_step=True):
    dones = tensordict_data[('next', 'done')]
    optimal_costs = tensordict_data[('next', 'optimal_cost')][dones]
    episode_rewards = tensordict_data[('next', 'episode_reward')][dones]
    optimality = (-optimal_costs / episode_rewards).mean().item()
    logs['episodes'] += dones.sum().item()
    if isinstance(optim, tuple):
        actor_lr = optim[0].param_groups[0]["lr"]
        critic_lr = optim[1].param_groups[0]["lr"]
    else:
        actor_lr = optim.param_groups[0]["lr"]
        critic_lr = optim.param_groups[1]["lr"]
    if logger is not None:
        logger.log(do_log_step=False, prefix="train", mean_episode_reward=episode_rewards.mean().item(),
                   optimality=optimality, episodes=logs['episodes'], collected_frames=logs['collected_frames'],
                   iteration=it)
        t_keys = tensordict_data.keys()
        if 'loc' in t_keys and 'scale' in t_keys:
            p = {'loc': wandb.Histogram(np_histogram=np.histogram(tensordict_data['loc'].reshape(-1, 1))),
                 'scale': wandb.Histogram(np_histogram=np.histogram(tensordict_data['scale'].reshape(-1, 1)))}
        else:
            p = {'param': wandb.Histogram(np_histogram=np.histogram(tensordict_data['param'].reshape(-1, 1)))}

        if 'mod_action' in t_keys:
            regrets = (optimal_costs + episode_rewards) / optimal_costs
            logger.log(do_log_step=False, prefix="debug", regret=regrets.mean().item(),
                       regret_values=wandb.Histogram(np_histogram=np.histogram(regrets)))
            action = tensordict_data[('next', 'mod_action')]
        else:
            action = tensordict_data['action'].reshape(-1, 1)
        logger.log(do_log_step=do_log_step, prefix="debug", actor_lr=actor_lr, critic_lr=critic_lr, **p,
                   action=wandb.Histogram(np_histogram=np.histogram(action)))
    pbar.update(tensordict_data.numel())
    epi_str = f"episodes seen: {logs['episodes']}"
    train_opt_str = f"train optimality: {optimality: 4.4f} "
    logs["act_lr"].append(actor_lr)
    logs["cri_lr"].append(critic_lr)
    lr_str = f"lr policy: {actor_lr: 4.4f}"
    return lr_str, epi_str, train_opt_str


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


def eval_and_log(cfg, logger, policy_module, test_env, prefix='eval'):
    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        # execute rollouts with the trained policy
        eval_rollouts = torch.stack([test_env.rollout(cfg.eval_rollout_steps, policy_module)
                                     for _ in range(cfg.eval_rollouts)])
    episode_rewards = eval_rollouts[('next', 'episode_reward')][:, -1]
    optimal_values = eval_rollouts[('next', 'optimal_cost')][:, -1]
    optimalities = - optimal_values / episode_rewards
    episode_reward = episode_rewards.mean().item()
    optimality = optimalities.mean().item()
    regrets = None
    if logger is not None:
        if cfg.data.problem == 'msc':
            regrets = (optimal_values + episode_rewards) / optimal_values
            logger.log(do_log_step=False, prefix=prefix, regret=regrets.mean().item(),
                       regret_std=regrets.std().item(),
                       regret_values=wandb.Histogram(np_histogram=np.histogram(regrets)))
            action = eval_rollouts[('next', 'mod_action')].reshape(cfg.eval_rollouts,
                                                                   test_env.action_space.shape[0])
        else:
            action = eval_rollouts['action'].reshape(cfg.eval_rollouts, test_env.n)
        logger.log(prefix=prefix, episode_reward=episode_reward, optimality=optimality,
                   episode_reward_std=episode_rewards.std(), optimality_std=optimalities.std(),
                   episode_reward_values=wandb.Histogram(
                       np_histogram=np.histogram(episode_rewards)),
                   optimality_values=wandb.Histogram(
                       np_histogram=np.histogram(optimalities)),
                   action=wandb.Histogram(np_histogram=np.histogram(action)))
    return {'episode_reward': episode_reward, 'optimality': optimality,
            'episode_rewards': episode_rewards, 'optimal_values': optimal_values,
            'optimalities': optimalities, 'regrets': regrets,
            'eval_rollouts': eval_rollouts}


def evaluate_policy(cfg, logger, logs, policy_module, test_env):
    info = eval_and_log(cfg, logger, policy_module, test_env)
    episode_reward, optimality = info['episode_reward'], info['optimality']
    logs["eval optimality"].append(optimality)
    logs["eval reward (sum)"].append(episode_reward)
    eval_str = (
        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
        f"eval optimality: {logs['eval optimality'][-1]: 4.4f} "
        f"(best: {logs['best_optimality']: 4.4f}), "
    )
    if optimality > logs['best_optimality']:
        logs['best_optimality'] = optimality
        logs['best_episodes'] = logs['episodes']
        logs['best_updates'] = logs['updates']
        name = get_model_name(cfg)
        dir_name = get_dir_name(cfg, logger)
        torch.save(policy_module.state_dict(), f'{dir_name}/{name}')
        if logger is not None:
            wandb.save(f'{dir_name}/{name}')
            logger.log(do_log_step=False, prefix="best", optimality=optimality,
                       episodes=logs['best_episodes'], updates=logs['best_updates'])

    test_env.reset()  # reset the env after the eval rollout
    eval_rollouts = info.pop('eval_rollouts')
    del eval_rollouts
    return eval_str


def final_evaluation(cfg, logger, policy_module, test_env, test_dir=None):
    name = get_model_name(cfg)
    dir_name = get_dir_name(cfg, logger) if test_dir is None else test_dir
    policy_module.load_state_dict(torch.load(f'{dir_name}/{name}'))
    policy_module.eval()  # might not be necessary

    info = eval_and_log(cfg, logger, policy_module, test_env, prefix='final_eval_stats')

    if cfg.data.problem == 'ems':
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
            last_episode_reward = info['episode_rewards'][-1].item()
            last_optimality = info['optimalities'][-1].item()
            logger.log(prefix='final_eval', episode_reward=last_episode_reward, optimality=last_optimality, **d,
                       opt_chart=axes[0].get_figure(), opt_chart_data=wandb.Table(dataframe=optimal_solution_df))
    else:  # msc
        if logger is not None:
            act_spec = test_env.action_space.shape[0]
            last_episode_reward = info['episode_rewards'][-1].item()
            last_optimality = info['optimalities'][-1].item()
            action = info['eval_rollouts'][('next', 'mod_action')][-1, 0]
            demands = info['eval_rollouts']['demands'][-1, 0]
            regret = info['regrets'][-1].item()
            t = wandb.Table(data=[[x, a, d, delta] for (x, a, d, delta) in zip(np.arange(act_spec),
                                                                               action, demands, demands - action)],
                            columns=["x", "action", 'demands', 'not_satisfied_demands'])
            logger.log(prefix='final_eval', episode_reward=last_episode_reward,
                       optimality=last_optimality, regret=regret,
                       action=wandb.plot.bar(t, "x", "action"),
                       demands=wandb.plot.bar(t, "x", "demands"),
                       not_satisfied_demands=wandb.plot.bar(t, "x", "not_satisfied_demands"))


def get_dir_name(cfg, logger=None):
    dn = logger.experiment.dir if logger is not None else f'outputs/{cfg.data.problem}/{cfg.model.policy}'
    if not os.path.exists(dn):
        os.makedirs(dn)
    return dn


def get_model_name(cfg):
    return f"best_{cfg.model.policy}_policy.pt"
