import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

class A2C(nn.Module):

    def __init__(self, input_shape, action_dim, core_nn, core_nn_output_size, world_model):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = action_dim
        self.core_nn = core_nn
        self.world_model = world_model
        self.policy_head = nn.Linear(core_nn_output_size, action_dim)
        self.value_head = nn.Linear(core_nn_output_size, 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4, eps=1e-7)

    def forward(self, x, valid_moves=None):
        if valid_moves is None:
            valid_moves = torch.ones(x.shape[0], 4)
        bottleneck = self.core_nn(x)
        policy_logits = self.policy_head(bottleneck)
        self._mask_invalid_actions(policy_logits, valid_moves)
        value = self.value_head(bottleneck)

        policy = torch.log_softmax(policy_logits, 1)
        return policy, value

    def select_actions(self, states, greedy):

        probs, _ = self.forward(states)
        probs = probs.detach().clone().cpu()
        dist = torch.distributions.categorical.Categorical(probs=probs)
        actions = dist.sample().numpy() if not greedy else torch.argmax(probs, 1).clone().numpy()
        action_prob = probs[range(len(actions)), actions]
        return actions, action_prob

    def _mask_invalid_actions(self, logits, valid_moves):
        valid_moves = torch.log(valid_moves)
        min_mask = torch.ones(*valid_moves.size(), dtype=torch.float) * torch.finfo(torch.float).min
        inf_mask = torch.max(valid_moves, min_mask).to(self.device)
        # Mask the logits of invalid actions with a number very close to the minimum float number.
        return logits + inf_mask

    def train_on_loader(self, loader: DataLoader):
        for batch in loader:
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, action_probs = batch

            rewards = rewards.to(self.device)
            states = self.world_model.encode(states.to(self.device)).detach()
            policy_log_probs, value = self.forward(states)

            adv = rewards - value.detach()

            log_probs = policy_log_probs[range(len(actions)), actions]

            current_probs = torch.exp(log_probs).clone().detach()
            importance_sample_ratio = current_probs / action_probs.to(self.device)

            policy_loss = (importance_sample_ratio.detach() * (-adv * log_probs)).mean()

            value = value.squeeze(1)
            value_loss = torch.nn.functional.mse_loss(value, rewards).mean()

            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            self.optimizer.step()