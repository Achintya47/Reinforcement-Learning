from gymnasium.utils.play import play
import gymnasium as gym

env = gym.make('LunarLander-v3', render_mode='rgb_array')
total_reward = 0

def step_callback(prev_obs, obs, action, reward, terminated, truncated, info):  
    global total_reward
    total_reward += reward
    if terminated or truncated:
        print(f"Episode Score: {total_reward:.2f}")
        total_reward = 0  # reset for next episode

play(env, keys_to_action={'w': 2, 'a': 1, 'd': 3}, noop=0, callback=step_callback)