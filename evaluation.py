import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs, make_sprites_state_env


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device, eval_reward_sums):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)
    # eval_envs = make_sprites_state_env(seed, )
    # 這裏改成我們自定義的environment，然後下面也要改

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        print('is not None')
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # while len(eval_episode_rewards) < 10:
    while len(eval_episode_rewards) < 40:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        # obs, _, done, infos = eval_envs.step(action)
        obs, rewards, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        # for info in infos:
        #     if 'episode' in info.keys():
        #         eval_episode_rewards.append(info['episode']['r'])

        for reward in rewards:
            eval_episode_rewards.append(reward)

    eval_envs.close()

    eval_reward_sums.append(np.sum(eval_episode_rewards).item())

    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    print(" Evaluation using {} episodes: mean reward {:.7f} sum reward {:.7f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.sum(eval_episode_rewards)))
