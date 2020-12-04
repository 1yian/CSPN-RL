import torch
import torch.nn as nn
import numpy as np

class WorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim, encoder, decoder, autoencoder, cspn):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.cspn = cspn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.encoder_iter = 0
        self.cspn_iter = 0
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-5, weight_decay=1e-7, eps=1e-7)
        self.cspn_optimizer = torch.optim.Adam(self.parameters(), lr=5e-6, weight_decay=1e-7, eps=1e-7)
    def forward(self, input):
        latent_state = self.autoencoder.encode(input)
        repr, reward = self.cspn.mpe(latent_state)
        prob = self.cspn.forward(latent_state)
        return repr, reward, prob

    def decode(self, latent_state):
        return self.decoder(latent_state)

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next_state(self, latent_state, action):
        latent_state = self.combine_latent_state_and_action(latent_state, action)
        rvs, reward = self.cspn.mpe(latent_state, True)
        repr = self.decode(rvs)
        return repr, reward, rvs

    def combine_latent_state_and_action(self, latent_state, action):
        latent_state = latent_state.view(-1, self.latent_dim)
        one_hot = torch.nn.functional.one_hot(action, num_classes=4).view(-1, 4)
        latent_state = torch.cat([latent_state, one_hot.to(self.device)], 1)
        return latent_state

    def train_on_loader(self, loader, writer):
        for batch in loader:
            self.autoencoder_optimizer.zero_grad()
            states, actions, rewards, _, next_states, action_probs = batch
            latent_states = self.encode(states).to(self.device)
            reconstructed = self.decode(latent_states)
            encoder_loss = (nn.MSELoss()(reconstructed, states.to(self.device)))
            writer.add_scalar('World_Model/Autoencoder', encoder_loss, self.encoder_iter)
            self.encoder_iter += 1
            encoder_loss.backward()
            self.autoencoder_optimizer.step()
        for batch in loader:
            self.cspn_optimizer.zero_grad()
            states, actions, rewards, _, next_states, action_probs = batch
            latent_states = self.encode(states).to(self.device)
            latent_states_actions = self.combine_latent_state_and_action(latent_states, actions)

            next_latent_states = self.encode(next_states.to(self.device))
            next_latent_states = torch.cat([next_latent_states, rewards.unsqueeze(-1).to(self.device)], 1)
            log_prob = self.cspn.forward(next_latent_states.detach(), latent_states_actions.detach())
            cspn_loss = (- log_prob * torch.exp(log_prob)).mean()
            writer.add_scalar('World_Model/CSPN', cspn_loss, self.cspn_iter)
            self.cspn_iter += 1
            cspn_loss.backward()
            self.cspn_optimizer.step()
