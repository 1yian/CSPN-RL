import rat_cspn
import torch
import torch.nn as nn
import region_graph
import numpy as np
import timeit

generic_nn = nn.Sequential(
    nn.Linear(256, 128),
    nn.Linear(128, 64),
)
rg = region_graph.RegionGraph(range(28))
for _ in range(0, 8):
    rg.random_split(2, 2)
cspn = rat_cspn.CSPN(rg)
cspn.make_cspn(generic_nn, 64)
test_input = torch.zeros((1, 28)).cuda()
test_conditional = torch.zeros((1, 256)).cuda()
print(test_input.dtype)
print(torch.exp(cspn.forward(test_input, test_conditional)))
max = torch.tensor(cspn.mpe(test_conditional), dtype=torch.float32).unsqueeze(0).cuda()
print(max)
print(torch.exp(cspn.forward(max, test_conditional)))

