import torch
import torch.nn as nn
import rat_cspn
import region_graph


class AutoEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.size = input_shape
        self.conv1 = nn.Conv2d(3, 16, 8, 1)
        self.conv2 = nn.Conv2d(16, 16, 8, 1)
        self.conv3 = nn.Conv2d(16, 8, 4, 1)
        self.conv4 = nn.Conv2d(8, 4, 4, 1)
        self.t_conv1 = nn.ConvTranspose2d(4, 8, 4)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 4)
        self.t_conv3 = nn.ConvTranspose2d(16, 16, 8)
        self.t_conv4 = nn.ConvTranspose2d(16, 3, 8)

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def encode(self, input):
        input = input.to(self.device)
        x = self.conv1(input)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)

        x = self.flatten(x)

        return x

    def decode(self, x):
        x = x.view(-1, 4, 12, 12)
        x = self.t_conv1(x)
        x = nn.ReLU()(x)
        x = self.t_conv2(x)
        x = nn.ReLU()(x)
        x = self.t_conv3(x)
        x = nn.ReLU()(x)
        x = self.t_conv4(x)
        x = nn.ReLU()(x)
        return x

    def forward(self, input):
        latent_state = self.encode(input)
        reconstruct = self.decode(latent_state)
        return reconstruct

class WorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        rg = region_graph.RegionGraph(range(latent_dim))
        for _ in range(0, 8):
            rg.random_split(2, 2)
        self.cspn = rat_cspn.CSPN(rg, latent_dim)

        self.autoencoder = AutoEncoder((3, 100, 100))
        generic_nn = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.cspn.make_cspn(generic_nn, 64)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-7, eps=1e-7)

    def forward(self, input):
        latent_state = self.autoencoder.encode(input)
        repr = self.cspn.mpe(latent_state)
        prob = self.cspn.forward(latent_state)
        return repr, prob

    def decode(self, latent_state):
        return self.autoencoder.decode(latent_state)

    def encode(self, obs):
        return self.autoencoder.encode(obs)

    def predict_next_state(self, latent_state, action):
        latent_state = self.combine_latent_state_and_action(latent_state, action)
        rvs = self.cspn.mpe(latent_state)
        repr = self.decode(rvs)
        return repr

    def combine_latent_state_and_action(self, latent_state, action):
        latent_state = latent_state.view(-1, self.latent_dim)
        action_onehot = torch.zeros(latent_state.shape[0], self.action_dim).to(self.device)
        action_onehot[range(len(action)), action] = 1
        latent_state = torch.cat([latent_state, action_onehot], 1)
        return latent_state


    def train_on_loader(self, loader):
        for batch in loader:
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, action_probs = batch
            latent_states = self.autoencoder.encode(states).to(self.device)
            latent_states_actions = self.combine_latent_state_and_action(latent_states, actions)

            next_latent_states = self.autoencoder.encode(next_states.to(self.device))
            cspn_loss = (- self.cspn.forward(next_latent_states.detach(), latent_states_actions.detach())).mean()

            reconstructed = self.autoencoder.decode(latent_states)
            encoder_loss = (nn.MSELoss()(reconstructed, states.to(self.device))).mean()

            loss = cspn_loss + encoder_loss
            loss.backward()
            self.optimizer.step()