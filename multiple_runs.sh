set +e
source .venv/bin/activate

python src/train.py --config-name  config_mlp_vector_stats \
    strategies.quad_width=0.5 strategies.quad_height=0.5 || echo "Failed to run config_mlp_vector_stats with quad_width=0.5 and quad_height=0.5"

python src/train.py --config-name  config_mlp_vector_stats \
    strategies.quad_width=0.25 strategies.quad_height=0.25 || echo "Failed to run config_mlp_vector_stats with quad_width=0.25 and quad_height=0.25"

python src/train.py --config-name  config_mlp_vector_stats \
    strategies.quad_width=0.125 strategies.quad_height=0.125 || echo "Failed to run config_mlp_vector_stats with quad_width=0.125 and quad_height=0.125"

python src/train.py --config-name  config_mlp_vector_handle_only || echo "Failed to run config_mlp_vector_handle_only"

python src/train.py --config-name  config_mlp_stats_sliding_window \
    strategies.quad_width=0.25 strategies.quad_height=0.25 || echo "Failed to run config_mlp_stats_sliding_window with quad_width=0.25 and quad_height=0.25"

python src/train.py --config-name config_stats_tcn_vector || echo "Failed to run config_stats_tcn_vector"
python src/train.py --config-name config_cnn_tcn_quat || echo "Failed to run config_cnn_tcn_quat"
