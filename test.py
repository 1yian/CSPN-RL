import gym
import numpy as np
import torch
import torch.nn as nn
import torchvision
from world_model import WorldModel
from a2c import CSPN_A2C
from replay import ExperienceReplayBuffer, collate_experiences
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import random
import breakout_test

NUM_EPOCHS = 12
NUM_HALLUCINATIONS = 200
NUM_ROLLOUTS = 10
MAX_ENV_STEPS = 100_000

env = gym.make('Breakout-v0')
writer = SummaryWriter()
resize = torchvision.transforms.Resize((42, 32))

agent = CSPN_A2C(breakout_test.LATENT_DIM, breakout_test.ACTION_DIM, breakout_test.PolicyNetwork(),
                 breakout_test.ValueNetwork(), continuous=breakout_test.CONTINOUS)
autoencoder = breakout_test.AutoEncoder(42, 32)
world_model_cspn = breakout_test.ForwardModelCSPN()
world_model = WorldModel(breakout_test.LATENT_DIM, 1, autoencoder.encode, autoencoder.decode, autoencoder,
                         world_model_cspn)

for epoch in range(NUM_EPOCHS):
    print("AT EPOCH:", epoch)

    starting_obs = []
    buffer = ExperienceReplayBuffer()
    for rollout_idx in range(NUM_ROLLOUTS):
        print("EPISODE #", rollout_idx)
        obs = env.reset()

        obs = np.rollaxis(obs, 2, 0)
        obs = resize(torch.Tensor(obs))
        obs = obs.unsqueeze(0) / 255.0
        episode = []
        starting_obs.append(obs)
        for i in range(MAX_ENV_STEPS):
            env.render()

            state = world_model.encode(obs)

            action, _, action_probs = agent.forward(state.detach(), sample=True)

            action = action.long().item()
            action_prob = action_probs.item()

            next_obs, reward, done, info = env.step(action)
            next_obs = np.rollaxis(next_obs, 2, 0)
            next_obs = resize(torch.Tensor(next_obs)).unsqueeze(0) / 255.0
            if done:
                next_obs = obs

            experience = {
                'state': obs,
                'action': action,
                'reward': reward,
                'next_state': next_obs,
                'action_prob': action_prob,
                'discounted_reward': 0,
            }
            episode.append(experience)
            obs = next_obs
            if done:
                break

        buffer.add(episode)

    loader = DataLoader(buffer, batch_size=128, shuffle=True, num_workers=8,
                        collate_fn=collate_experiences)

    world_model.train_on_loader(loader, writer)
    agent.train_on_loader(loader, writer, autoencoder.encode)
    obs = random.choice(starting_obs)
    latent_state = world_model.encode(obs)
    action = torch.LongTensor([0])
    pic = world_model.predict_next_state(latent_state, action)[0].cpu()

    grid = torchvision.utils.make_grid(torch.cat([obs, pic], 0))
    writer.add_image('image_{}'.format(epoch), grid)
    buffer.clear()
    print("Dreaming...", epoch)
    for dream in range(NUM_HALLUCINATIONS):
        print("HALLUCINATION #", dream)
        obs = random.choice(starting_obs)
        state = world_model.encode(obs)
        episode = []
        for i in range(25):
            action, _, action_probs = agent.forward(state.detach(), sample=True)

            action = action.long().item()
            action_prob = action_probs.item()

            _, reward, next_state = world_model.predict_next_state(state, torch.LongTensor([action]))

            experience = {
                'state': state.detach().cpu(),
                'action': action,
                'reward': reward.detach().item(),
                'next_state': next_state.detach().cpu(),
                'action_prob': action_prob,
                'discounted_reward': 0,
            }

            state = next_state

            episode.append(experience)
        buffer.add(episode)

    loader = DataLoader(buffer, batch_size=256, shuffle=True, num_workers=8,
                        collate_fn=collate_experiences)

    agent.train_on_loader(loader, writer, None)

'''
for iter in range(20):
    print("iter", iter)
    for i_episode in range(50):
        obs = env.reset()
        obs = np.rollaxis(obs, 2, 0)
        obs = resize(torch.Tensor(obs)).unsqueeze(0) / 255.0

        episode = []
        for t in range(100):

            env.render()
            state = world_model.encode(obs)
            action, action_prob = agent.select_actions(state.detach(), False)
            action = action.item()
            action_prob = action_prob.item()
            next_obs, reward, done, info = env.step(action)
            next_obs = np.rollaxis(next_obs, 2, 0)
            next_obs = resize(torch.Tensor(next_obs)).unsqueeze(0) / 255.0
            if done:
                next_obs = obs
            experience = {
                'state': obs,
                'action': action,
                'reward': reward,
                'next_state': next_obs,
                'action_prob': action_prob
            }
            episode.append(experience)
            obs = next_obs
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        buffer.add(episode)
    env.close()
    loader = DataLoader(buffer, batch_size=64, shuffle=True, num_workers=10,
                        collate_fn=collate_experiences)
    print("training agent...")
    agent.train_on_loader(loader)
    print("training world model...")
    world_model.train_on_loader(loader)
    obs = env.reset()
    obs = np.rollaxis(obs, 2, 0)
    obs = resize(torch.Tensor(obs)).unsqueeze(0)
    latent_state = world_model.encode(obs.cuda())
    action = torch.LongTensor([0])
    print("getting a sample image\n")
    pic = world_model.predict_next_state(latent_state, action)[0]
    save_image(pic, 'img{}.png'.format(iter))
    buffer.clear()
'''
