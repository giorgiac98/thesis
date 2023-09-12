from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig, omegaconf
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from utils import env_maker, prepare_networks_and_policy, training_loop, MyWandbLogger, final_evaluation


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device = hydra.utils.instantiate(cfg.device)
    if cfg.update_rounds * cfg.batch_size >= cfg.frames_per_batch:
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_params = dict(config=wandb_cfg,
                            project="thesis-experiments",
                            group=cfg.data.problem,
                            tags=[cfg.data.problem, cfg.model.policy,
                                  str(cfg.data.params.instance), cfg.data.params.method])
        logger = MyWandbLogger(exp_name=None, **wandb_params) if cfg.wandb_log else None
        env_maker_function = env_maker(**cfg.data, device=device)

        test_env = env_maker_function(logger)
        env_state_dict = test_env.transform[0].state_dict()
        env_action_spec, env_obs_spec = test_env.action_spec, test_env.observation_spec
        policy_module, loss_module, other = prepare_networks_and_policy(**cfg.model,
                                                                        device=device,
                                                                        env_action_spec=env_action_spec,
                                                                        input_shape=env_obs_spec['observation'].shape[0])
        total_frames = cfg.frames_per_batch * cfg.train_iter
        collector = MultiSyncDataCollector(frames_per_batch=cfg.frames_per_batch,
                                           create_env_fn=[env_maker_function] * cfg.num_envs,
                                           create_env_kwargs=[{'state_dict': env_state_dict}] * cfg.num_envs,
                                           policy=policy_module,
                                           total_frames=total_frames,
                                           device=device)
        buffer_size = cfg.frames_per_batch if cfg.model.policy == 'ppo' else cfg.buffer_size
        replay_buffer = TensorDictReplayBuffer(
            batch_size=cfg.batch_size,
            storage=LazyMemmapStorage(buffer_size),
            sampler=hydra.utils.instantiate(cfg.model.other_spec.sampler)
        )
        optim = torch.optim.Adam(loss_module.parameters(), cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // cfg.frames_per_batch, cfg.end_lr
        ) if cfg.schedule_lr else None
        # test_env.do_log(True)
        training_loop(cfg, policy_module, loss_module, other, optim, collector, replay_buffer, device, test_env,
                      logger=logger, scheduler=scheduler)
        # final evaluation and comparison with the oracle
        if cfg.data.problem == 'ems':  # TODO: implement for msc
            final_evaluation(cfg, logger, policy_module, test_env)


if __name__ == "__main__":
    main()
