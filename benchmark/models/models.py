"""
FileName: models.py
Description: GNN models' set
Time: 2020/7/30 9:01
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from benchmark.args import data_args
from torch_geometric.data.batch import Batch
from benchmark.models.utils import ReadOut
from torch_geometric.nn import TopKPooling,GlobalAttention

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor


class GNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch

class GCN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return out


class GIN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

class GIN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCNConv(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out


class GINConv(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.edge_weight = None
        self.fc_steps = None
        self.reweight = None

    # def children(self):
    #     if
    #     return iter([])


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if data_args.task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)


        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)

class GlobalMaxPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_max_pool(x, batch)

class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_add_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x


class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

class GGNN_simplify(nn.Module):
    def __init__(self, output_dim=200, num_steps=6):
        super().__init__()
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = gnn.GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.readout = GlobalMaxPool()
        self.classifier = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()
        #elf.sigmoid = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.relu(self.ggnn(x, edge_index))
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        avg = self.classifier(pooled)
        result = self.sigmoid(avg)
        return result

class GCN_simplify2(nn.Module):

    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        num_layer = 3
        self.out_dim = output_dim #200
        self.in_dim = input_dim
        self.conv1 = gnn.GCNConv(input_dim, output_dim)
        self.conv2 = gnn.GCNConv(output_dim, output_dim)
        self.conv3 = gnn.GCNConv(output_dim, output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)##0.3
        self.connect = nn.Linear(output_dim, output_dim)
        self.readout = GlobalMaxPool()
        self.__classifier = nn.Linear(output_dim, 2)
        self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv,edge_index))
        post_conv = self.conv3(post_conv,edge_index)
        pooled = self.readout(post_conv, torch.zeros(post_conv.shape[0], dtype=int, device=post_conv.device))
        #pooled = self.readout(post_conv, batch)
        y_a = self.__classifier(pooled)
        result = self.softmax(y_a)
        return result

class GraphConvEncoder(torch.nn.Module):
    """

    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)

    """

    def __init__(self, output_dim=200, input_dim=100):
        super(GraphConvEncoder, self).__init__()

        self.input_GCL = gnn.GCNConv(input_dim, output_dim)

        self.input_GPL = TopKPooling(output_dim,
                                     ratio=0.8)

        self.attpool = GlobalAttention(torch.nn.Linear(output_dim, 1))

    def forward(self,x, edge_index):
        # [n nodes; rnn hidden]
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        node_embedding = F.relu(self.input_GCL(x, edge_index))
        node_embedding, _, _, batch, _, _ = self.input_GPL(node_embedding, edge_index)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)
        return out

class DeepWukong(nn.Module):
    def __init__(self, output_dim=200, input_dim=100):
        super().__init__()
        # self.conv = gnn.GCNConv(input_dim, output_dim)
        self.conv = GraphConvEncoder(output_dim, input_dim)
        hidden_size=2*output_dim
        #hidden_size = 256
        layers = [
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)##0.5
        ]
        layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)##0..5
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, 2)
        #self.readout = GlobalAddPool()
        self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.Sigmoid()
    def forward(self, x, edge_index):
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.conv(x, edge_index)
        # pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        hiddens = self.__hidden_layers(outputs)
        avg = self.__classifier(hiddens) 
        result = self.softmax(avg)             
        return result

class ExtractFeature(nn.Module):
    def __init__(self,input_dim=200, out_dim=400, num_layers=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2)##0.2
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=input_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) for _ in range(num_layers)])
    def forward(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

class RevealModel(nn.Module):
    def __init__(self,input_dim=100, output_dim=200, num_steps=6):##6
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.hidden_dim = 400
        self.num_timesteps = num_steps
        self.num_layers = 1
        self.relu = nn.ReLU()
        self.readout = GlobalAddPool()
        self.ggnn = gnn.GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.extract_feature=ExtractFeature()
        # self.__classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # self.softmax=nn.Softmax(dim=1)
        self.__classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            # nn.Softmax(dim=-1)
            nn.Sigmoid()
        )

    
    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.relu(self.ggnn(x, edge_index))
        pooled = self.readout(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        #pooled = self.readout(outputs, batch)
        h_a = self.extract_feature(pooled)
        y_a = self.__classifier(h_a)
        #result = self.softmax(y_a)
        return y_a

class DevignModel(nn.Module):
    def __init__(self,input_dim=100, output_dim=200, num_steps=6):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = gnn.GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)   # [1,100,4]
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def de_batchify_graphs(self, features=None):
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)

        vectors = [torch.tensor(1)]
        vectors[0] = torch.tensor(features,requires_grad = True)
        output_vectors = torch.stack(vectors)

        return output_vectors

    def get_network_inputs(self, graph, cuda=False, device=None):
        features = graph.ndata['features']
        edge_types = graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return graph, features, edge_types
        pass

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.ggnn(x, edge_index)
        x_i = self.de_batchify_graphs(x)
        h_i = self.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        Y_1 = self.maxpool1(
            self.relu(
                # self.conv_l1(h_i.transpose(1, 2))  # num_node >= 5
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2)) #outputs
                )
            )
        )
        Y_2 = self.maxpool2(
            self.relu(
                # self.conv_l2(Y_1)
                self.batchnorm_1d(
                    self.conv_l2(Y_1) #outputs
                )
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            self.relu(
                # self.conv_l1_for_concat(c_i.transpose(1, 2))
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2)) #ouputs+feature
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            self.relu(
                # self.conv_l2_for_concat(Z_1)
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1) #ouputs+feature
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg)
        return result
    

class DevignModel_mod(nn.Module):
    def __init__(self,input_dim=100, output_dim=200, num_steps=8):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim #200
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = gnn.GatedGraphConv(out_channels=output_dim, num_layers=num_steps)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 4)   # [1,100,4]
        self.maxpool1 = torch.nn.MaxPool1d(4, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 2)
        self.maxpool2 = torch.nn.MaxPool1d(3, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 4)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(4, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 2)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(3, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Softmax()

    def de_batchify_graphs(self, features=None):
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)

        vectors = [torch.tensor(1)]
        vectors[0] = torch.tensor(features,requires_grad = True)
        output_vectors = torch.stack(vectors)

        return output_vectors

    def get_network_inputs(self, graph, cuda=False, device=None):
        features = graph.ndata['features']
        edge_types = graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return graph, features, edge_types
        pass

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.ggnn(x, edge_index)
        x_i = self.de_batchify_graphs(x)
        h_i = self.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        Y_1 = self.maxpool1(
            self.relu(
                # self.conv_l1(h_i.transpose(1, 2))  # num_node >= 5
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2)) #outputs
                )
            )
        )
        Y_2 = self.maxpool2(
            self.relu(
                # self.conv_l2(Y_1)
                self.batchnorm_1d(
                    self.conv_l2(Y_1) #outputs
                )
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            self.relu(
                # self.conv_l1_for_concat(c_i.transpose(1, 2))
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2)) #ouputs+feature
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            self.relu(
                # self.conv_l2_for_concat(Z_1)
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1) #ouputs+feature
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg)
        return result