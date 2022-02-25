# Reinforcement Learning with Dual-Observation
The implementation of Reinforcement Learning with Dual-Observation technique.

The basic reinfocement learning algorithm is adapted from [stable-baseline3](https://github.com/DLR-RM/stable-baselines3)

### Requirments:

```
pip install -r requirements.txt
```
### Run
Follow the basci running command. Refer to `./arguments.py` for more options
```
python train.py --algo PPO --total-timesteps 1000000 --env-name golddigger
```


### Note:
Please modify following variables according you own setting in `./environment/GOLOEnv.py`.
```
da_environment_path 
gvgai_path 
```
