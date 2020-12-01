import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


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

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4, eps=1e-7)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, inputs, sample=False):
        """
        Not meant to be used outside of this class/object.
        @param inputs: the batch of input tensors to predict
        @param sample: a bool on whether we should pick stochastically or greedily
        @return: if self.contiuous is true, we return the desired values of each continuous action
                else return a vector that gives log probabilities of picking each discrete action.
        """
        if sample:
            policy = self.policy_head.sample(inputs)
            value = self.value_head.sample(inputs)
        else:
            policy = self.policy_head.forward_mpe(inputs)
            value = self.value_head.forward_mpe(inputs)

        # In the continuous case, we return the raw values predicted or sampled by our CSPN.
        # In the discrete case, we can log softmax the logits given by our CSPN so that the probs are normalized.
        if not self.continuous:
            policy = torch.log_softmax(policy, 1)
        return policy, value

    def select_actions(self, inputs, sample=False):
        if self.continuous:
            actions, values = self.forward(inputs, sample)
            probs = self.policy_head.forward(inputs)
        else:
            probs, values = self.forward(inputs, sample)
            probs = probs.detach().clone().cpu()
            dist = torch.distributions.categorical.Categorical(probs=probs)
            actions = dist.sample().numpy() if not sample else torch.argmax(probs, 1).clone().numpy()
            probs = probs[range(len(actions)), actions]
        return actions, probs, values

    def train_on_loader(self, loader: DataLoader):
        for batch in loader:
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, action_probs = batch

            rewards = rewards.to(self.device)
            _, cur_log_probs, values = self.select_actions(states)

            adv = rewards - values.detach()

            current_probs = torch.exp(cur_log_probs).clone().detach()
            importance_sample_ratio = current_probs / action_probs.to(self.device)

            policy_loss = (importance_sample_ratio.detach() * (-adv * cur_log_probs)).sum()

            values = values.squeeze(1)
            value_loss = torch.nn.functional.mse_loss(values, rewards).sum()

            total_loss = policy_loss + value_loss
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()