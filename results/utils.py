
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import List

import wandb
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

DEFAULT_STORE_KEYS = ['data.problem', 'model.policy',
                      'frames_per_batch', 'num_envs', 'update_rounds', 'batch_size', 'buffer_size', 'prb',
                      'total_frames', 'actor_lr', 'critic_lr', 'schedule_lr', 'model.actor_net_spec.num_cells',
                      'model.value_net_spec.num_cells', 'data.params.obs_norm', ]

DEFAULT_METRICS = ['best/optimality', 'best/episodes', 'best/updates', 'eval/optimality']


def get_from_wandb(run, store_keys, summary_keys=None):
    """Download run data from wandb.
    :param run: wandb.Run
    :param store_keys: tuple of str; the keys to store in the dataframe.
    :param summary_keys: list of str; the keys to be retrieved through summary method.
    :return: str, dict; the variant and the metrics used to compute intervals for this run.
    """

    def unwrap_config(k):
        cfg = run.config
        for k in k.split('.'):
            cfg = cfg[k]
        if isinstance(cfg, str) and cfg.startswith('${.'):
            cfg = run.config[cfg[3:-1]]
        if isinstance(cfg, str):
            cfg = cfg.upper()
        elif isinstance(cfg, list):
            cfg = f"{len(cfg)} instance" + ('s' if len(cfg) > 1 else '')

        return cfg

    if run.config['data']['problem'] == 'ems':
        store_keys += ('data.params.instances', 'data.params.method')
    hyperparams = {k: unwrap_config(k) for k in [*store_keys]}
    d = dict()
    for sk in summary_keys:
        s = run.summary[sk]
        if isinstance(s, wandb.old.summary.SummarySubDict):
            s = s['max']
        d[sk] = s
    return {**hyperparams, **d}


def compute_ci(x):
    """Compute bootstrap confidence interval for a given array of values."""
    ci = bootstrap([x], statistic=np.mean, random_state=42).confidence_interval
    return ci.low, ci.high


def sanitize(s, sep='.'):
    return s.split(sep)[-1]


def get_save_path(prefix, sweep_ids):
    return f"{prefix}/{'-'.join(sweep_ids)}"


def load_data(save_path: str,
              sweep_ids: List[str],
              store_keys: List[str],
              metrics: List[str],
              force_download: bool = False):
    """
    Load data from wandb or from a local file.
    :param save_path: str; the path where to save the data.
    :param sweep_ids: list of str; the sweep ids to download.
    :param store_keys: list of str; the keys to store in the dataframe that are not in DEFAULT_STORE_KEYS.
    :param metrics: list of str; the metrics to store in the dataframe that are not in DEFAULT_METRICS.
    :param force_download: bool; if True, download the data from wandb even if it is already present in the save_path.
            Default is False.
    :return: pd.DataFrame; the data.
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isfile(f'{save_path}/data.csv') or force_download:
        api = wandb.Api()
        sweeps = [api.sweep(f'giorgiac98/thesis/{s_id}') for s_id in sweep_ids]
        runs = [r for sw in sweeps for r in sw.runs
                if sw.id != '5nr9v4xd' or sw.id == '5nr9v4xd' and r.config['frames_per_batch'] <= 240]
        sk = store_keys + DEFAULT_STORE_KEYS
        metrics += DEFAULT_METRICS
        map_f = partial(get_from_wandb, store_keys=sk, summary_keys=metrics)
        with ThreadPoolExecutor(max_workers=100) as ex:
            print(f'Downloading data from wandb (Sweep ID: {sweep_ids})')
            raw_data = ex.map(map_f, runs)
        df = pd.DataFrame(list(raw_data))
        if 'final_eval_stats/regret' in df.columns:
            df['final_eval_stats/regret'] = -df['final_eval_stats/regret']
        df.to_csv(f'{save_path}/data.csv', index=False)
        print('Data saved to', f'{save_path}/data.csv')
    else:
        print(f'Loading data from {save_path}/data.csv')
        df = pd.read_csv(f'{save_path}/data.csv')
    tf_map = {20000: '20K', 40000: '40K', 60000: '60K', 1000000: '1M', 37000: '37K',
              480000: '480K', 288000: '288K', 182400: '182K', 96000: '96K'}
    df['total_frames'] = df['total_frames'].apply(lambda x: tf_map[x])
    return df
