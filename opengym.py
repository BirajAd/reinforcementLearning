import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np

# for i in range(200):
#   action = env.action_space.sample()
#   env.step(action)
#   env.render()

def playOneGame(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
      left_proba = model(obs[np.newaxis])
      action = (tf.random.uniform([1,1]) > left_proba)
      y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
      loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads

# playOneGame(env, env.reset(), model, loss_fn)

def playMultipleEpisodes(env, episodes, n_steps, model, loss_fn):
  all_rewards = []
  all_grads = []
  for ep in range(episodes):
    current_rewards = []
    current_grads = []
    obs = env.reset()
    for step in range(n_steps):
      obs, reward, done, grads = playOneGame(env, obs, model, loss_fn)
      current_rewards.append(reward)
      current_grads.append(grads)
      if done:
        break
    all_rewards.append(current_rewards)
    all_grads.append(current_grads)
  return all_rewards, all_grads


def discountRewards(rewards, dis_rate):
  discount = np.array(rewards)
  for step in range(len(rewards)-2, -1,-1):
    discount[step] += discount[step+1] * dis_rate
  return discount

def discountAndNormalizeRewards(all_rewards, dis_rate):
  all_discounted_rewards = [discountRewards(reward, dis_rate) for reward in all_rewards]
  flat_rewards = np.concatenate(all_discounted_rewards)
  reward_mean = flat_rewards.mean()
  reward_std = flat_rewards.std()
  
  return [(reward - reward_mean)/reward_std for reward in all_discounted_rewards]

total_iterations = 150
episode_per_update = 10
max_steps = 200
discount_rate = 0.95

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[4]),
    keras.layers.Dense(1, activation="sigmoid"),
])

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(42)

for iteration in range(total_iterations):
  all_rewards, all_grads = playMultipleEpisodes(env, episode_per_update, max_steps, model, loss_fn)
  total_rewards = sum(map(sum, all_rewards))
  print("\rIteration: {}, mean rewards: {:.1f}".format(iteration, total_rewards/episode_per_update),end="")
  final_rewards = discountAndNormalizeRewards(all_rewards, discount_rate)

  all_mean_grads = []
  for var_index in range(len(model.trainable_variables)):
    mean_grads = tf.reduce_mean([reward * all_grads[episode_index][step][var_index] for episode_index, rewards in enumerate(final_rewards) for step, reward in enumerate(rewards)], axis=0)
    all_mean_grads.append(mean_grads)
  optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

env.close()

import matplotlib.pyplot as plt
from matplotlib import animation

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames

frames = render_policy_net(model)
plot_animation(frames)