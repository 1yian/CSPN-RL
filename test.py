import gym
import numpy as np
import torch
import torch.nn as nn
import torchvision
from world_model import WorldModel
from a2c import A2C
from replay import ExperienceReplayBuffer, collate_experiences
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
env = gym.make('Breakout-v0')

world_model = WorldModel(576, 4)

agent_core = nn.Sequential(
    nn.Linear(576, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
)
buffer = ExperienceReplayBuffer()
resize = torchvision.transforms.Resize((32, 32))
agent = A2C((576), 4, agent_core, 64, world_model)
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


