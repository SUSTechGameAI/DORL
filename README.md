# Reinforcement Learning with Dual-Observation
This is the implementation of the paper "Reinforcement Learning with Dual-Observation for General Video Game Playing".

The basic reinfocement learning algorithm is adapted from [stable-baseline3](https://github.com/DLR-RM/stable-baselines3).

Check [Generic Video Game Competition (GVGAI) Learning framework](https://github.com/SUSTechGameAI/GVGAI_GYM) from game environment.
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
