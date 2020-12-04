import region_graph
import rat_cspn
import a2c
import replay
import torch
import torch.nn as nn

LATENT_DIM = 128
ACTION_DIM = 4

A2C_CONDITIONAL_NN_BOTTLENECK_DIM = 64
POLICY_CSPN_DEPTH = 4
POLICY_CSPN_NUM_RECURSIONS = 12
CONTINOUS = False

VALUE_CSPN_DEPTH = 4
VALUE_CSPN_NUM_RECURSIONS = 12

FORWARD_MODEL_CONDITIONAL_NN_BOTTLENECK_DIM = 128
ENCODER_INPUT = 256
FORWARD_MODEL_CSPN_DEPTH = 12
FORWARD_MODEL_CSPN_NUM_RECURSIONS = 12


class A2CConditionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(LATENT_DIM, 256)
        self.fc2 = nn.Linear(256, A2C_CONDITIONAL_NN_BOTTLENECK_DIM)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_graph = region_graph.RegionGraph(range(1))
        for i in range(POLICY_CSPN_NUM_RECURSIONS):
            self.region_graph.random_split(1, POLICY_CSPN_DEPTH // 2)

        self.cspn = rat_cspn.CSPN(self.region_graph, 1, A2CConditionalNN(), A2C_CONDITIONAL_NN_BOTTLENECK_DIM,
                                  continuous=CONTINOUS, rv_domain=range(ACTION_DIM))

    def forward_mpe(self, input, sample):

        return self.cspn.mpe(input, sample)

    def forward(self, action, input):
        return self.cspn.forward(action, input)


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_graph = region_graph.RegionGraph(range(1))
        for i in range(VALUE_CSPN_NUM_RECURSIONS):
            self.region_graph.random_split(1, VALUE_CSPN_DEPTH // 2)

        self.cspn = rat_cspn.CSPN(self.region_graph, 1, A2CConditionalNN(), A2C_CONDITIONAL_NN_BOTTLENECK_DIM,
                                  continuous=True)

    def forward_mpe(self, input, sample):
        return self.cspn.mpe(input, sample)

    def forward(self, value, input):
        return self.cspn.forward(value, input)


class ForwardModelConditionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(LATENT_DIM + ACTION_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, FORWARD_MODEL_CONDITIONAL_NN_BOTTLENECK_DIM)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.size = (width, height)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3 * self.size[0] * self.size[1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, LATENT_DIM)

        self.fc5 = nn.Linear(LATENT_DIM, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, 3 * self.size[0] * self.size[1])


        self.flatten = torch.nn.Flatten(start_dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, input):
        input = input.to(self.device)
        x = self.flatten(input)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = self.flatten(x)

        return x

    def decode(self, x):
        x = x.to(self.device)
        x = self.fc5(x)
        x = nn.ReLU()(x)
        x = self.fc6(x)
        x = nn.ReLU()(x)
        x = self.fc7(x)
        x = nn.ReLU()(x)
        x = self.fc8(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 3, self.size[0], self.size[1])
        return x

    def forward(self, input):
        latent_state = self.encode(input)
        reconstruct = self.decode(latent_state)
        return reconstruct


class ForwardModelCSPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_graph = region_graph.RegionGraph(range(LATENT_DIM + 1))
        for i in range(FORWARD_MODEL_CSPN_NUM_RECURSIONS):
            self.region_graph.random_split(2, FORWARD_MODEL_CSPN_DEPTH // 2)

        self.cspn = rat_cspn.CSPN(self.region_graph, LATENT_DIM + 1, ForwardModelConditionalNN(),
                                  FORWARD_MODEL_CONDITIONAL_NN_BOTTLENECK_DIM, continuous=True)

    def mpe(self, input, sample):
        result = self.cspn.mpe(input, sample)
        return result[:, :-1], result[:, -1]

    def forward(self, input, conditional):
        return self.cspn.forward(input, conditional)
