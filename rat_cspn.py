import torch
import torch.nn as nn
import torch.distributions as dists
import region_graph
import enum
import numpy as np


class CSPN(nn.Module):
    def __init__(self, region_graph):
        super().__init__()
        self.num_sums = 2
        self.num_leaves = 2

        self.num_dims = region_graph.get_num_items()

        self.region_graph = region_graph

        # Map the regions to its log-probability tensor
        self.region_distributions = dict()

        # Map the regions to a partition- a list of children log-prob tensors
        self.region_products = dict()

        self.tensor_list = nn.ModuleList()
        self.output_tensor = None
        self.region_graph_layers = None
        self.parameter_nn = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def make_cspn(self, nn_core, nn_output_size):
        self.region_graph_layers = self.region_graph.make_layers()
        id_counter = 0
        # Make the leaf layer.
        leaf_layer = nn.ModuleList()
        for leaf_region in self.region_graph_layers[0]:
            leaf_tensor = GaussianTensor(leaf_region, id=id_counter)
            id_counter += 1
            leaf_layer.append(leaf_tensor)
            self.region_distributions[leaf_region] = leaf_tensor

        self.tensor_list.append(leaf_layer)

        def add_to_map(given_map, key, item):
            existing_items = given_map.get(key, [])
            given_map[key] = existing_items + [item]

        for layer_idx in range(1, len(self.region_graph_layers)):
            layer = nn.ModuleList()
            if layer_idx % 2 == 1:
                partitions = self.region_graph_layers[layer_idx]
                for partition in partitions:
                    input_regions = list(partition)
                    input1 = self.region_distributions[input_regions[0]]
                    input2 = self.region_distributions[input_regions[1]]
                    product_tensor = ProductTensor(input1, input2, id=id_counter)
                    id_counter += 1
                    layer.append(product_tensor)

                    resulting_region = tuple(sorted(input_regions[0] + input_regions[1]))
                    add_to_map(self.region_products, resulting_region, product_tensor)

            else:
                num_sums = self.num_sums if layer_idx != len(self.region_graph_layers) - 1 else 1
                regions = self.region_graph_layers[layer_idx]

                for region in regions:
                    product_tensors = self.region_products[region]
                    sum_tensor = GatingTensor(product_tensors, num_sums, id=id_counter)
                    id_counter += 1
                    layer.append(sum_tensor)
                    self.region_distributions[region] = sum_tensor

            self.tensor_list.append(layer)
        self.output_tensor = self.region_distributions[self.region_graph.get_root_region()]
        self.parameter_nn = ParameterNN(self.tensor_list, nn_core, nn_output_size)

    def mpe(self, conditional):
        self.forward_mpe(conditional)
        random_variables = np.zeros()
        self.output_tensor.backwards_mpe(random_variables, 0)

    def forward_mpe(self, conditional):
        obj_to_tensor = {}
        obj_to_params = self.parameter_nn.forward(conditional)

        for leaf_tensor_obj in self.tensor_list[0]:
            params = obj_to_params[leaf_tensor_obj]
            obj_to_tensor[leaf_tensor_obj] = leaf_tensor_obj.forward(params, params)

        for layer_idx in range(1, len(self.tensor_list)):
            for tensor_obj in self.tensor_list[layer_idx]:
                input_tensors = [obj_to_tensor[obj] for obj in tensor_obj.inputs]
                params = obj_to_params[tensor_obj]
                output_tensor = tensor_obj.forward(input_tensors, params=params)
                obj_to_tensor[tensor_obj] = output_tensor
        return obj_to_tensor[self.output_tensor]

    def forward(self, inputs, conditional, marginalized=None):
        obj_to_tensor = {}
        obj_to_params = self.parameter_nn.forward(conditional)

        for leaf_tensor_obj in self.tensor_list[0]:
            params = obj_to_params[leaf_tensor_obj]
            obj_to_tensor[leaf_tensor_obj] = leaf_tensor_obj.forward(inputs, params, marginalized)

        for layer_idx in range(1, len(self.tensor_list)):
            for tensor_obj in self.tensor_list[layer_idx]:
                input_tensors = [obj_to_tensor[obj] for obj in tensor_obj.inputs]
                params = obj_to_params[tensor_obj]
                output_tensor = tensor_obj.forward(input_tensors, params=params)
                obj_to_tensor[tensor_obj] = output_tensor
        return obj_to_tensor[self.output_tensor]


class GaussianTensor(nn.Module):
    def __init__(self, region, id):
        super().__init__()
        self.id = id
        self.size = 2
        self.scope = sorted(list(region))
        self.num_gauss = len(self.scope)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.scope_tensor = torch.LongTensor(self.scope).to(self.device)

    def forward(self, inputs, params, marginalized=None):
        means = params
        dist = dists.Normal(means, torch.ones(means.shape).to(self.device))

        local_inputs = torch.index_select(inputs, 1, self.scope_tensor).unsqueeze(-1)
        # Using the log trick so we're working on the log domain
        log_pdf = torch.sum(dist.log_prob(local_inputs), 1)
        self.dist_argmax = means
        return log_pdf

    def backwards_mpe(self, random_variables, node_idx):
        rv = self.scope_tensor[node_idx].detach().cpu()
        random_variables[rv] = self.dist_argmax[0][node_idx]

    def type(self):
        return NodeTensorType.gaussian

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        return self.id == other.id


class GatingTensor(nn.Module):
    def __init__(self, product_tensors, num_sums, id):
        super().__init__()
        self.inputs = product_tensors

        self.id = id
        self.size = num_sums
        self.scope = self.inputs[0].scope

        self.num_inputs = 0
        for inp in self.inputs:
            assert set(inp.scope) == set(self.scope)
            self.num_inputs += inp.size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, inputs, params, marginalized=None):
        # Using the log trick so we're working on the log domain
        weights = torch.log_softmax(params, 1)
        prods = torch.cat(inputs, 1)
        child_values = prods.unsqueeze(-1) + weights
        self.max_child_idxs = torch.argmax(child_values, axis=1)
        sums = torch.logsumexp(child_values, 1)
        return sums

    def backward_mpe(self, random_vars, node_idx):
        max_child_idx = self.max_child_idxs[0][node_idx]
        prod_tensor_idx = max_child_idx // len(self.inputs)
        node_idx = max_child_idx % self.inputs[prod_tensor_idx].size
        prod_tensor_idx[prod_tensor_idx].backward_mpe(random_vars, node_idx)

    def type(self):
        return NodeTensorType.gated

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        return self.id == other.id


class ProductTensor(nn.Module):
    def __init__(self, tensor_obj1, tensor_obj2, id):
        super().__init__()
        self.id = id

        self.inputs = [tensor_obj1, tensor_obj2]

        self.child_idx_map = self.make_child_map(tensor_obj1.size, tensor_obj2.size)
        self.scope = list(set(tensor_obj1.scope) | set(tensor_obj2.scope))
        self.size = tensor_obj1.size * tensor_obj2.size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, inputs, params, marginalized=None):
        dists1 = inputs[0]
        dists2 = inputs[1]
        batch_size = dists1.shape[0]
        num_dist1 = int(dists1.shape[1])
        num_dist2 = int(dists2.shape[1])

        # we take the outer product via broadcasting, thus expand in different dims
        dists1_expand = dists1.unsqueeze(1)
        dists2_expand = dists2.unsqueeze(2)

        # Using the log trick so we're working on the log domain
        prod = dists1_expand + dists2_expand
        # flatten out the outer product
        prod = prod.view([batch_size, num_dist1 * num_dist2])
        return prod

    def backward_mpe(self, random_vars, node_idx):
        node = self.child_idx_map[node_idx]
        self.inputs[0].backward_mpe(random_vars, node[0])
        self.inputs[1].backward_mpe(random_vars, node[1])

    def make_child_map(self, size1, size2):
        indices = []
        for i in range(size1):
            for j in range(size2):
                indices.append((i, j))
        return indices

    def type(self):
        return NodeTensorType.product

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        return self.id == other.id


class NodeTensorType(enum.Enum):
    gaussian = 0
    product = 1
    gated = 2


class ParamType(enum.Enum):
    mean = 0
    sigma = 1
    gated = 2


class ParameterNN(nn.Module):
    def __init__(self, tensor_obj_layers, core, core_output_size):
        super().__init__()
        self.param_providers = {}
        self.param_objs = nn.ModuleList()
        for tensor_obj_layer in tensor_obj_layers:
            for tensor_obj in tensor_obj_layer:
                type = tensor_obj.type()
                if type == NodeTensorType.gaussian:
                    param_provider = ParamProvider(ParamType.mean, core_output_size, tensor_obj.size)
                elif type == NodeTensorType.gated:
                    param_provider = ParamProvider(ParamType.gated, core_output_size,
                                                   tensor_obj.num_inputs * tensor_obj.size)
                else:
                    param_provider = None

                self.param_providers[tensor_obj] = param_provider
                self.param_objs.append(param_provider)

        # Core can be a generic NN that has output size of (batch_size, core_output_size)
        self.core = core
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input):
        bottleneck = self.core.forward(input)
        outputs = {}
        for obj in self.param_providers:
            param_provider = self.param_providers[obj]
            type = obj.type()
            params = None
            if param_provider:

                params = param_provider.forward(bottleneck)
                if type == NodeTensorType.gaussian:
                    params = params.view(-1, param_provider.output_size)
                elif type == NodeTensorType.gated:
                    params = params.view(-1, obj.num_inputs, obj.size)
            else:
                assert (type == NodeTensorType.product)
            outputs[obj] = params
        return outputs


class ParamProvider(nn.Module):
    def __init__(self, param_type, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.param_type = param_type
        self.raw_param_layer = nn.Linear(input_size, output_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input):
        output = self.raw_param_layer(input)
        if self.param_type == ParamType.mean:
            pass
        elif self.param_type == ParamType.sigma:
            pass
        elif self.param_type == ParamType.gated:
            pass
        return output
