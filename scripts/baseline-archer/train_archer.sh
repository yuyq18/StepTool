
export ARCHER_CONFIG_NAME="archer_config.yaml"

accelerate launch --config_file config/archer/accelerate_config.yaml src/baseline-archer/run.py