import copy
import random
from collections import deque
from random import randrange

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_sparse import SparseTensor

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class SOGA(BaseAdapter):
    """
    SOGA from "Source Free Graph Unsupervised Domain Adaptation (WSDM 2024)".
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        super().__init__(pre_model, source_stats, adapter_config)
        self.epochs = adapter_config.epochs
        self.learning_rate = adapter_config.lr
        self.struct_lambda = adapter_config.struct_lambda
        self.neigh_lambda = adapter_config.neigh_lambda
        self.num_negative_samples = 5
        self.num_positive_samples = 2

    def adapt(self, data: Data) -> Tensor:
        self.model.to(self.device)
        data = data.to(self.device)
        target_data = Data(x=data.x, edge_index=data.edge_index)
        target_structure_data = target_data.clone()
        row, col = data.edge_index
        self.num_target_nodes = data.x.size(0)
        structure_adj = SparseTensor(
            row=row,
            col=col,
            sparse_sizes=(self.num_target_nodes, self.num_target_nodes),
        )
        self.init_target(target_structure_data, target_data)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            self.model.train()
            _, output = self.model(data)
            probs = F.softmax(output, dim=-1)

            NCE_loss_struct = self.NCE_loss(
                probs,
                self.center_nodes_struct,
                self.positive_samples_struct,
                self.negative_samples_struct,
            )
            NCE_loss_neigh = self.NCE_loss(
                probs,
                self.center_nodes_neigh,
                self.positive_samples_neigh,
                self.negative_samples_neigh,
            )

            IM_loss = self.ent(probs) - self.div(probs)

            loss = (
                IM_loss
                + self.struct_lambda * NCE_loss_struct
                + self.neigh_lambda * NCE_loss_neigh
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        _, output = self.model(data)
        probs = F.softmax(output, dim=-1)
        return probs

    def init_target(self, graph_struct, graph_neigh):
        self.target_G_struct = to_networkx(graph_struct)
        self.target_G_neigh = to_networkx(graph_neigh)

        self.Positive_Sampler = RandomWalker(
            self.target_G_struct, p=0.25, q=2, use_rejection_sampling=1
        )
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_struct, self.positive_samples_struct = (
            self.generate_positive_samples()
        )
        self.negative_samples_struct = self.generate_negative_samples()

        self.Positive_Sampler = RandomWalker(
            self.target_G_neigh, p=0.25, q=2, use_rejection_sampling=1
        )
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_neigh, self.positive_samples_neigh = (
            self.generate_positive_samples()
        )
        self.negative_samples_neigh = self.generate_negative_samples()

    def NCE_loss(self, outputs, center_nodes, positive_samples, negative_samples):
        negative_embedding = F.embedding(negative_samples, outputs)
        positive_embedding = F.embedding(positive_samples, outputs)
        center_embedding = F.embedding(center_nodes, outputs)

        positive_embedding = positive_embedding.permute([0, 2, 1])
        positive_score = torch.bmm(center_embedding, positive_embedding).squeeze()
        exp_positive_score = torch.exp(positive_score).squeeze()

        negative_embedding = negative_embedding.permute([0, 2, 1])
        negative_score = torch.bmm(center_embedding, negative_embedding).squeeze()
        exp_negative_score = torch.exp(negative_score).squeeze()

        exp_negative_score = torch.sum(exp_negative_score, dim=1)

        loss = -torch.log(exp_positive_score / exp_negative_score)
        loss = loss.mean()

        return loss

    def generate_positive_samples(self):
        self.Positive_Sampler.preprocess_transition_probs()
        self.positive_samples = self.Positive_Sampler.simulate_walks(
            num_walks=1, walk_length=self.num_positive_samples, workers=1, verbose=1
        )
        for i in range(len(self.positive_samples)):
            if len(self.positive_samples[i]) != 2:
                self.positive_samples[i].append(self.positive_samples[i][0])

        samples = torch.tensor(self.positive_samples).to(self.device)

        center_nodes = torch.unsqueeze(samples[:, 0], dim=-1)
        positive_samples = torch.unsqueeze(samples[:, 1], dim=-1)

        return center_nodes, positive_samples

    def generate_negative_samples(self):
        negative_samples = (
            torch.tensor(
                [
                    self.Negative_Sampler.sample()
                    for _ in range(self.num_negative_samples * self.num_target_nodes)
                ]
            )
            .view([self.num_target_nodes, self.num_negative_samples])
            .to(self.device)
        )

        return negative_samples

    def ent(self, softmax_output):
        entropy_loss = torch.mean(Entropy(softmax_output))

        return entropy_loss

    def div(self, softmax_output):
        mean_softmax_output = softmax_output.mean(dim=0)
        diversity_loss = torch.sum(
            -mean_softmax_output * torch.log(mean_softmax_output + 1e-8)
        )

        return diversity_loss


def Entropy(input):
    batch_size, num_feature = input.size()
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[
                        alias_sample(alias_edges[edge][0], alias_edges[edge][1])
                    ]
                    walk.append(next_node)
            else:
                break

        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if inv_p > second_upper_bound:
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs)
                    )
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if prob + shatter >= upper_bound:
                            next_node = prev
                            break
                        next_node = cur_nbrs[
                            alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])
                        ]
                        if prob < lower_bound:
                            break
                        if prob < inv_p and next_node == prev:
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if prob < _prob:
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = self._simulate_walks(nodes, num_walks, walk_length)

        # Parallel(n_jobs=workers, verbose=verbose, )(
        #     delayed(self._simulate_walks)(nodes, num, walk_length) for num in
        #     partition_num(num_walks, workers))

        # walks = list(itertools.chain(*results))

        return results

    def _simulate_walks(
        self,
        nodes,
        num_walks,
        walk_length,
    ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(
                        self.deepwalk_walk(walk_length=walk_length, start_node=v)
                    )
                elif self.use_rejection_sampling:
                    walks.append(
                        self.node2vec_walk2(walk_length=walk_length, start_node=v)
                    )
                else:
                    walks.append(
                        self.node2vec_walk(walk_length=walk_length, start_node=v)
                    )  # [1:]
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get("weight", 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                G[node][nbr].get("weight", 1.0) for nbr in G.neighbors(node)
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(
                        edge[1], edge[0]
                    )
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return


class Negative_Sampler:
    def __init__(self, G):
        self.G = G
        _probs = [G.degree(i) for i in G.nodes()]
        probs = np.array(_probs, dtype=np.float64)
        self.num = len(_probs)
        probs = probs / np.sum(probs)
        # print probs
        self.probs_table = np.ones(self.num, dtype=np.float64)
        self.bi_table = np.zeros(self.num, dtype=np.int32)
        p = 1.0 / self.num
        L, H = [], []
        # 按照是否大于平均水平进行区分
        for i in range(self.num):
            if probs[i] < p:
                L.append(i)
            else:
                H.append(i)

        while len(L) > 0 and len(H) > 0:
            # 把一个序列排清楚
            l = L.pop()
            h = H.pop()
            self.probs_table[l] = probs[l] / p
            self.bi_table[l] = h
            probs[h] = probs[h] - (p - probs[l])
            if probs[h] < p:
                L.append(h)
            else:
                H.append(h)
        del L, H
        # print self.probs_table
        # print self.bi_table

    def sample(self):
        idx = randrange(self.num)
        if random.random() < self.probs_table[idx]:
            return idx
        else:
            return self.bi_table[idx]

    def construct_graph_origin(self, G):
        new_G = nx.Graph()
        new_G.graph["degree"] = 0
        dq = deque()
        for iter in range(self.num_walks):
            for u in G.nodes():
                dq.clear()
                dq.append(u)
                v = u
                if v not in new_G:
                    new_G.add_node(v)
                    new_G.node[v]["degree"] = 0
                for t in range(self.walk_length):
                    adj = list(G[v])
                    v_id = random.randint(0, len(adj) - 1)
                    v = adj[v_id]
                    if v not in new_G:
                        new_G.add_node(v)
                        new_G.node[v]["degree"] = 0
                    for it in dq:
                        if it in new_G[v]:
                            new_G[v][it]["weight"] += 1
                        else:
                            new_G.add_edge(v, it, weight=1)
                        new_G.graph["degree"] += 1
                        new_G.node[v]["degree"] += 1
                        new_G.node[it]["degree"] += 1
                    dq.append(v)
                    if len(dq) > self.window_size:
                        dq.popleft()
        return new_G


def create_alias_table(area_ratio):
    """

    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """

    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]
