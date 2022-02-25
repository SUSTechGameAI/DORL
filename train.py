from arguments import get_args
args = get_args()
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import gym
import torch
print(torch.cuda.device_count())
torch.cuda.set_device(int(args.gpuid))
import gym_gvgai as gvg
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.dqn.policies import DQNPolicy

from environment.GOLOEnv import GOLOEnv
import sys
import numpy as np
from networks import SingleConvExtractor, DoubleInputConvExatractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
np.set_printoptions(threshold=sys.maxsize)




if __name__ == '__main__':
    if args.use_one_hot:
        if args.use_local_observation:
            folder = "onehot_golo/"
        else:
            folder = "onehot_go/"
    else:
        if args.use_local_observation:
            folder = "img_golo/"
        else:
            folder = "img_go/"


    folder += args.env_name +"/"
    log_path = args.log_dir + folder
    save_path = args.save_dir + folder
    print(save_path)
    model_name = args.algo+"_"+args.env_name+"_"+"onehot_"+str(args.use_one_hot)+"_lo_"+str(args.use_local_observation)
    print("model name",model_name)
    assert args.algo in ["DQN", "A2C", "PPO"]
    save_path = os.path.join(save_path,model_name)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=save_path)
    event_callback = EveryNTimesteps(n_steps=100000, callback=checkpoint_on_event)
    

    if args.algo == "DQN":
        env = GOLOEnv(args.env_name, args.use_one_hot, args.use_local_observation,args.algo)
    else:
        print(args.algo)
        env = make_vec_env(
            GOLOEnv, n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"game": args.env_name, "use_one_hot": args.use_one_hot, "use_LO": args.use_local_observation,"algorithm":args.algo})
    if args.use_local_observation:
        extractor_cls = DoubleInputConvExatractor
    else:
        extractor_cls = SingleConvExtractor

    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": {'one_hot': args.use_one_hot},
        'net_arch': []
    }
    if args.algo == "DQN":
        agent = DQN(
            DQNPolicy, env, buffer_size=40000, verbose=1,
            learning_starts=0, policy_kwargs=policy_kwargs, tensorboard_log=log_path
        )
    elif args.algo == "A2C":
        agent = A2C(
            ActorCriticPolicy, env, verbose=1,
            policy_kwargs=policy_kwargs, tensorboard_log=log_path
        )
    elif args.algo == "PPO":
        agent = PPO(
            ActorCriticPolicy, env, verbose=1,
            policy_kwargs=policy_kwargs, tensorboard_log=log_path
        )
    print(args.total_timesteps)
    agent.save(save_path+"/rl_model_0")
    #default log_interval=1000
    agent.learn(total_timesteps=args.total_timesteps, callback=event_callback, log_interval=100)
    agent.save(save_path + "final" + args.algo)


