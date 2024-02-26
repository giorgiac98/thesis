# ppo-msc
python hp_plots.py o5p5qkhs
# sac-msc
python hp_plots.py 54m5e4is --variant-keys prb batch_size actor_lr model.other_spec.alpha_lr frames_per_batch model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets --store-keys model.other_spec.alpha_lr model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak
# td3-msc
python hp_plots.py y2k6fq22 71t85fes --variant-keys prb batch_size buffer_size prb frames_per_batch buffer_size frames_per_batch model.other_spec.init_random_frames buffer_size model.other_spec.init_random_frames update_rounds model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets --store-keys model.other_spec.init_random_frames model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak

# ppo-ems-seq
python hp_plots.py 23bhohyf
# sac-ems-seq
python hp_plots.py 5nr9v4xd q2ob5ove tj8mtfq9 --variant-keys prb batch_size actor_lr model.other_spec.alpha_lr frames_per_batch model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak update_rounds model.other_spec.target_update_polyak model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets --store-keys model.other_spec.alpha_lr model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak
# td3-ems-seq
python hp_plots.py fea9ghlv vjyvoean --variant-keys prb batch_size buffer_size prb frames_per_batch buffer_size frames_per_batch model.other_spec.init_random_frames buffer_size model.other_spec.init_random_frames update_rounds model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets model.policy_spec.loss_function optim --store-keys model.other_spec.init_random_frames model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.policy_spec.loss_function optim

# ppo-ems-ss
python hp_plots.py tst7yhva
# sac-ems-ss
python hp_plots.py gqbh7srs --variant-keys prb batch_size actor_lr model.other_spec.alpha_lr frames_per_batch model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets --store-keys model.other_spec.alpha_lr model.other_spec.init_random_frames model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak
# td3-ems-ss
python hp_plots.py i93evk7x --variant-keys prb batch_size buffer_size prb frames_per_batch buffer_size frames_per_batch model.other_spec.init_random_frames buffer_size model.other_spec.init_random_frames update_rounds model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.policy_spec.loss_function optim model.value_net_spec.num_cells model.policy_spec.num_qvalue_nets --store-keys model.other_spec.init_random_frames model.other_spec.policy_delay_update model.policy_spec.num_qvalue_nets model.other_spec.target_update_polyak model.policy_spec.loss_function optim
