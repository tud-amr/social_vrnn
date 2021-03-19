echo "Run 1"
python3 train.py --exp_num 300 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6
echo "Run 2"
python3 train.py --exp_num 300 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6
echo "Run 3"
python3 train.py --exp_num 300 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6
echo "Run 4"
python3 train.py --exp_num 300 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6
echo "Run 5"
python3 train.py --exp_num 300 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6