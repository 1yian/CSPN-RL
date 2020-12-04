import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np

class CSPN_A2C(nn.Module):

    def __init__(self, input_shape, action_dim, policy_head, value_head, continuous=False):
        super().__init__()

        # Is this A2C acting in a discrete or continuous action space?
        self.continuous = continuous

        # The shape of the input space. For the purpose of this experiment,
        # the A2C will take the latent single dimensional vector as its observation.
        self.input_shape = input_shape

        # The output shape will be the action_dim.
        # Will be one hot-encoded vector in the discrete case, or actual values in the continuous case
        self.output_shape = action_dim

        # These are our function approximations of the value and policy functions, most likely using a CSPN.
        # Calling forward or sample on these should give us a batch our the predicted/sampled policies and values
        # for every input
        self.policy_head = policy_head
        self.value_head = value_head

        self.iter = 0
        self.dreamer_iter = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-7, eps=1e-7)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, inputs, sample=False):
        """
        Not meant to be used outside of this class/object.
        @param inputs: the batch of input tensors to predict
        @param sample: a bool on whether we should pick stochastically or greedily
        @return: the values of the inputs that the policy picks, the value of the state, probs: the corresponding
        """
        policy = self.policy_head.forward_mpe(inputs, sample)
        value = self.value_head.forward_mpe(inputs, False)

        probs = torch.exp(self.policy_head.forward(policy, inputs))
        return policy, value, probs

    def train_on_loader(self, loader: DataLoader, writer, encoder=None):
        for batch in loader:
            self.optimizer.zero_grad()
            states, actions, _, rewards, next_states, action_probs = batch

            rewards = rewards.to(self.device)
            if encoder is not None:
                states = encoder(states).detach()
            states = states.to(self.device).detach()
            values = self.value_head.forward_mpe(states, False)
            cur_log_probs = torch.exp(self.policy_head.forward(actions, states))
            adv = rewards - values.detach()

            current_probs = torch.exp(cur_log_probs).clone().detach()
            importance_sample_ratio = current_probs / action_probs.to(self.device)

            policy_loss = (importance_sample_ratio.detach() * (-adv * cur_log_probs)).mean()
            values = values.squeeze(1)
            value_loss = torch.nn.functional.mse_loss(values, rewards).mean()

            total_loss = policy_loss.mean() + value_loss.mean()
            if encoder is not None:
                writer.add_scalar('A2C/Policy', policy_loss, self.iter)
                writer.add_scalar('A2C/Value', value_loss, self.iter)
                self.iter += 1
            else:
                writer.add_scalar('Dreamer/Policy', policy_loss, self.dreamer_iter)
                writer.add_scalar('Dreamer/Value', value_loss, self.dreamer_iter)
                self.dreamer_iter += 1
            total_loss.backward()
            self.optimizer.step()
