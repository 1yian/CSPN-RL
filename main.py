import rat_cspn
import torch
import torch.nn as nn
import region_graph
import numpy as np

generic_nn = nn.Sequential(
    nn.Linear(28, 128),
    nn.Linear(128, 64),
)
rg = region_graph.RegionGraph(range(28))
for _ in range(0, 8):
    rg.random_split(2, 2)
cspn = rat_cspn.CSPN(rg)
cspn.make_cspn(generic_nn, 64)
test = torch.zeros((1, 28))
print(cspn.state_dict().keys())
print(torch.exp(cspn.forward(test)))
