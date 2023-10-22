import hydra
import torch
from omegaconf import DictConfig, omegaconf
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers import PrioritizedSampler

from utils import env_maker, prepare_networks_and_policy, training_loop, MyWandbLogger, final_evaluation, set_seeds, \
    define_metrics, make_optimizer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device = hydra.utils.instantiate(cfg.device)
    set_seeds(cfg.seed)
    size = cfg.update_rounds * cfg.batch_size
    print(size)
    if cfg.model.policy != 'ppo' and size >= cfg.frames_per_batch or cfg.model.policy == 'ppo':
        if cfg.frames_per_batch % cfg.num_envs != 0:
            cfg.frames_per_batch = cfg.num_envs * (cfg.frames_per_batch // cfg.num_envs)
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_params = dict(config=wandb_cfg,
                            project="thesis-experiments",
                            group=cfg.data.problem,
                            tags=[cfg.data.problem, cfg.model.policy,
                                  str(cfg.data.params.instance), cfg.data.params.method])
        logger = MyWandbLogger(**wandb_params) if cfg.wandb_log else None
        if logger:
            define_metrics(cfg, logger)
        env_maker_function = env_maker(**cfg.data, device=device)

        test_env = env_maker_function(logger)
        env_state_dict = test_env.transform[0].state_dict()
        env_action_spec, env_obs_spec = test_env.action_spec, test_env.observation_spec
        policy_module, loss_module, other = prepare_networks_and_policy(**cfg.model,
                                                                        device=device,
                                                                        env_action_spec=env_action_spec,
                                                                        input_shape=env_obs_spec['observation'].shape[
                                                                            0])
        total_frames = cfg.frames_per_batch * cfg.train_iter
        exploration_policy = policy_module if cfg.model.policy != 'td3' else other['actor_model_explore']
        collector = MultiSyncDataCollector(frames_per_batch=cfg.frames_per_batch,
                                           create_env_fn=[env_maker_function] * cfg.num_envs,
                                           create_env_kwargs=[{'state_dict': env_state_dict}] * cfg.num_envs,
                                           policy=exploration_policy,
                                           total_frames=total_frames,
                                           device=device)
        collector.set_seed(cfg.seed)
        buffer_size = cfg.frames_per_batch if cfg.model.policy == 'ppo' else cfg.buffer_size
        if cfg.prb:
            sampler = PrioritizedSampler(buffer_size, alpha=0.7, beta=0.5)
        else:
            sampler = hydra.utils.instantiate(cfg.model.other_spec.sampler)
        replay_buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(
                                                   buffer_size,
                                                   scratch_dir='/tmp/' if cfg.model.policy != 'ppo' else None,
                                                   device=device,
                                               ),
                                               batch_size=cfg.batch_size,
                                               sampler=sampler)
        optim, scheduler = make_optimizer(cfg, loss_module)
        # test_env.do_log(True)
        training_loop(cfg, policy_module, loss_module, other, optim, collector, replay_buffer, device, test_env,
                      logger=logger, scheduler=scheduler)
        # final evaluation and comparison with the oracle
        if cfg.data.problem == 'ems':  # TODO: implement for msc
            final_evaluation(cfg, logger, policy_module, test_env)


if __name__ == "__main__":
    main()
