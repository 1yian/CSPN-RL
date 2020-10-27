import rat_cspn
import torch
import torch.nn as nn
import region_graph

generic_nn = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.Linear(128, 64),
)
rg = region_graph.RegionGraph(range(28 * 28))
for _ in range(0, 8):
    rg.random_split(2, 2)
cspn = rat_cspn.CSPN(rg)
cspn.make_cspn(generic_nn, 64)
test = torch.zeros((0, 28 * 28))
cspn.forward(test)
