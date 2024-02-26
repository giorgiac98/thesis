from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, compute_ci, sanitize, get_save_path


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('sweep_id', nargs='+')
    parser.add_argument('--variant-keys', nargs='+', default=list())
    parser.add_argument('--metrics', nargs='+', default=list())
    parser.add_argument('--store-keys', nargs='+', default=list())
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('Running with args:\n', args)
    sweep_id = args.sweep_id
    store_keys = args.store_keys
    variant_keys = args.variant_keys
    metrics = args.metrics
    save_path = get_save_path("sweep_hp_tuning", sweep_id)
    df = load_data(save_path, sweep_id, store_keys, metrics)

    keys = [('frames_per_batch', 'update_rounds'), ('update_rounds', 'batch_size'), ('batch_size', None),
            ('num_envs', None), ('num_envs', 'frames_per_batch'), ('actor_lr', 'critic_lr'),
            ('actor_lr', 'schedule_lr'), ('critic_lr', 'schedule_lr'),
            ('model.actor_net_spec.num_cells', 'model.value_net_spec.num_cells'),
            ('actor_lr', 'model.actor_net_spec.num_cells'),
            ('critic_lr', 'model.value_net_spec.num_cells'), ]
    keys += list(zip(variant_keys[::2], variant_keys[1::2]))
    keys += [(k, None) for k in store_keys]
    if df['data.problem'].iloc[0] == 'EMS':
        keys += [('data.params.instances', 'data.params.obs_norm')]
    else:
        keys += [('data.params.obs_norm', None)]
    if df['model.policy'].iloc[0] != 'PPO':
        keys += [('buffer_size', None), ('prb', None)]
    metric = 'eval/optimality.max'
    for variant, hue in keys:
        if hue is not None:
            hue_vals = df[hue].unique()
            assert len(hue_vals) > 1, f'There is only one unique val for {hue}, cannot use it as hue'
            palette = dict(zip(hue_vals, sns.color_palette(palette='deep', n_colors=len(hue_vals))))
            plot_kwargs = {'palette': palette, 'hue': hue, 'dodge': 0.7}
        else:
            plot_kwargs = {}
        col = None
        if variant == 'actor_lr' and hue == 'critic_lr':
            col = 'schedule_lr'
        cplot = sns.catplot(df, x=variant, y=metric, kind='point', linestyles='',
                            errorbar=lambda x: compute_ci(x), col=col, **plot_kwargs, capsize=0.1)
        cplot.fig.subplots_adjust(top=.95)
        fig_name = f'{sanitize(variant)}' if hue is None else f'{sanitize(variant)}_vs_{sanitize(hue)}'
        cplot.fig.savefig(f'{save_path}/{fig_name}.png')
        plt.close(cplot.fig)

    df_up = df[df['best/updates'] > 0]
    jplt = sns.jointplot(data=df_up, x="best/updates", y='eval/optimality.max', hue="model.actor_net_spec.num_cells",
                         palette='deep')
    x_col = df_up['best/updates']
    jplt.ax_joint.set_xlim([x_col.min() - 0.5, x_col.max() + 0.5])
    jplt.fig.savefig(f'{save_path}/updates_vs_optimality_actor_num_cells.png')
    jplt.ax_joint.set_xscale('log')
    jplt.fig.savefig(f'{save_path}/log_updates_vs_optimality_actor_num_cells.png')
