# Reinforcement Learning with Dual-Observation
This is the implementation of the paper "Reinforcement Learning with Dual-Observation for General Video Game Playing" accepted by the IEEE Transactions on Games in 2022. The basic reinfocement learning algorithm is adapted from [stable-baseline3](https://github.com/DLR-RM/stable-baselines3).

Please use this bibtex if you use this repository in your work:

````
@article{hu2022reinforcement,
  title={Reinforcement Learning with Dual-Observation for General Video Game Playing},
  author={Hu, Chengpeng and Wang, Ziqi and Shu, Tianye and Tong, Hao and Togelius, Julian and Yao, Xin and Liu, Jialin},
  journal={IEEE Transactions on Games},
  pages={accepted},
  year={2022},
  publisher={IEEE}
}
````

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
