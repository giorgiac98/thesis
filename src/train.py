import shutil

import hydra
import torch
from omegaconf import DictConfig, omegaconf
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers import PrioritizedSampler

from utils import env_maker, prepare_networks_and_policy, training_loop, MyWandbLogger, final_evaluation, set_seeds, \
    define_metrics, make_optimizer, get_dir_name


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device = hydra.utils.instantiate(cfg.device)
    set_seeds(cfg.seed)
    if cfg.frames_per_batch % cfg.num_envs != 0:
        cfg.frames_per_batch = cfg.num_envs * (cfg.frames_per_batch // cfg.num_envs)
        print(f'Warning: frames_per_batch is not divisible by num_envs --> new frames_per_batch:{cfg.frames_per_batch}')
    wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_params = dict(config=wandb_cfg,
                        project="thesis",
                        group=cfg.data.problem,
                        tags=[cfg.data.problem, cfg.model.policy])
    if cfg.data.problem == 'ems':
        len_instances = len(cfg.data.params.instances)
        if len_instances == 1:
            wandb_params['tags'] += [str(cfg.data.params.instances[0])]
        else:
            wandb_params['tags'] += [', '.join(map(lambda x: str(x), cfg.data.params.instances))]
        wandb_params['tags'] += [f'{len_instances} instance' + ('s' if len_instances > 1 else ''),
                                 cfg.data.params.method]
    else:
        wandb_params['tags'] += [f'{cfg.data.params.num_prods} prods x {cfg.data.params.num_sets} sets',
                                 f'{cfg.data.params.num_instances} instances']
    logger = MyWandbLogger(**wandb_params) if cfg.wandb_log else None
    if logger:
        define_metrics(cfg, logger)
    env_maker_function = env_maker(cfg.data.problem, cfg.data.params, device=device, seed=cfg.seed)

    garbage_env = env_maker_function()
    env_state_dict = garbage_env.transform[0].state_dict() if cfg.data.params.obs_norm else None
    env_action_spec, env_obs_spec = garbage_env.action_spec, garbage_env.observation_spec
    policy_module, loss_module, other = prepare_networks_and_policy(**cfg.model,
                                                                    device=device,
                                                                    problem_spec=cfg.data.problem_spec,
                                                                    env_action_spec=env_action_spec,
                                                                    input_shape=env_obs_spec['observation'].shape[
                                                                        0])
    exploration_policy = policy_module if cfg.model.policy != 'td3' else other['actor_model_explore']
    init_random_frames = cfg.model.other_spec.init_random_frames if 'init_random_frames' in cfg.model.other_spec else None
    collector = MultiSyncDataCollector(frames_per_batch=cfg.frames_per_batch,
                                       create_env_fn=[env_maker_function] * cfg.num_envs,
                                       create_env_kwargs=[{'state_dict': env_state_dict}] * cfg.num_envs,
                                       policy=exploration_policy,
                                       init_random_frames=init_random_frames,
                                       total_frames=cfg.total_frames,
                                       device=device)
    collector.set_seed(cfg.seed)
    buffer_size = cfg.frames_per_batch if cfg.model.policy == 'ppo' else cfg.buffer_size
    if cfg.prb and cfg.model.policy != 'ppo':
        sampler = PrioritizedSampler(buffer_size, alpha=0.7, beta=0.5)
    else:
        sampler = hydra.utils.instantiate(cfg.model.other_spec.sampler)
    replay_buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(
        buffer_size,
        scratch_dir='/tmp/',
        device=device,
    ),
        batch_size=cfg.batch_size,
        sampler=sampler)
    optim, scheduler = make_optimizer(cfg, loss_module)
    valid_env = env_maker_function(state_dict=env_state_dict, env_type='valid', num_test_instances=cfg.eval_rollouts)
    training_loop(cfg, policy_module, loss_module, other, optim, collector, replay_buffer, device, valid_env,
                  logger=logger, scheduler=scheduler)

    # final evaluation and comparison with the optimal solution
    test_env = env_maker_function(state_dict=env_state_dict, env_type='test', num_test_instances=cfg.test_rollouts)
    final_evaluation(cfg, logger, policy_module, test_env)

    dir_name = get_dir_name(cfg, logger)
    wandb_dir = dir_name[:-6] if 'files' in dir_name else dir_name
    if logger is not None:
        logger.experiment.finish()
    print(f'Cleaning up {wandb_dir}')
    shutil.rmtree(wandb_dir)


if __name__ == "__main__":
    main()
