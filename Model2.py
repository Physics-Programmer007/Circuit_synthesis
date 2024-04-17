import torch
import torch.nn as nn
import torch.nn.functional as F
import igraph


class CktGNN(nn.Module):
    # topology and node feature together.
    def __init__(self, max_n, nvt, subn_nvt, START_TYPE, END_TYPE, max_pos=8, emb_dim=16, feat_emb_dim=8, hs=301, nz=56,
                 bidirectional=False, pos=True, scale=True, scale_factor=102, topo_feat_scale=0.01):
        super(CktGNN, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.max_pos = max_pos + 1  # number of  positions in amp: 1 sudo + 7 positions
        self.scale = scale
        self.scale_factor = scale_factor
        self.nvt = nvt  # number of device types
        self.subn_nvt = subn_nvt + 1
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.emb_dim = emb_dim
        self.feat_emb_dim = feat_emb_dim  # continuous feature embedding dimension
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs + feat_emb_dim  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.pos = pos  # whether to use the prior knowledge
        self.topo_feat_scale = topo_feat_scale  # balance the attention to topology information
        self.device = None

        #
        if self.pos:
            self.vs = hs + self.max_pos  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.df_enc = nn.Sequential(
            nn.Linear(self.max_pos * 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, feat_emb_dim)
        )  # subg features can be canonized according to the position of subg

        self.grue_forward = nn.GRUCell(nvt + self.max_pos, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt + self.max_pos, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar

        # 1. decoding-related
        self.grud = nn.GRUCell(nvt + self.max_pos, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, nvt)
        )  # which type of new subg to add
        self.add_edge = nn.Sequential(
            nn.Linear(hs * 2 + self.max_pos * 2, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, 1)
        )  # whether to add edge between v_i and v_new
        self.add_pos = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, self.max_pos)
        )  # which position of new subg to add
        self.df_fc = nn.Sequential(
            nn.Linear(hs, 64),
            nn.ReLU(),
            nn.Linear(64, self.max_pos * 3)
        )  # decode subg features

        # 2. gate-related
        self.gate_forward = nn.Sequential(
            nn.Linear(self.vs, hs),
            nn.Sigmoid()
        )
        self.gate_backward = nn.Sequential(
            nn.Linear(self.vs, hs),
            nn.Sigmoid()
        )
        self.mapper_forward = nn.Sequential(
            nn.Linear(self.vs, hs, bias=False),
        )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
            nn.Linear(self.vs, hs, bias=False),
        )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                nn.Linear(hs * 2, hs),
            )
            self.hg_unify = nn.Sequential(
                nn.Linear(self.gs * 2, self.gs),
            )

        # 4. attention mechanism
        self.attention_linear = nn.Linear(hs * 2, 1)  # Linear layer for attention mechanism
        self.attention_softmax = nn.Softmax(dim=1)

        # 5. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device())  # get a zero hidden state

    def _get_zero_hidden(self, n=1, prior_edge=False):
        if prior_edge:
            return self._get_zeros(n, self.hs + self.max_pos)  # get a zero hidden state
        else:
            return self._get_zeros(n, self.hs)  # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False, decode=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:  # H: previous hidden state
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        pos_feats = [g.vs[v]['pos'] for g in G]
        X_v_ = self._one_hot(v_types, self.nvt)
        X_pos_ = self._one_hot(pos_feats, self.max_pos)

        X = torch.cat([X_v_, X_pos_], dim=1)

        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]  # hidden state of 'predecessors'
            if self.pos:
                pos_ = [self._one_hot([g.vs[v_]['pos'] for v_ in g.successors(v)], self.max_pos) for g in
                        G]  # one hot of vertex index of 'predecessors', pos_=vids
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.pos:
                pos_ = [self._one_hot([g.vs[x]['pos'] for x in g.predecessors(v)], self.max_pos) for g in
                        G]  # one hot of vertex index of 'predecessors', pos_=vids
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.pos:
            H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, pos_)]
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:  ### start point
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred +
                                    [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                          for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i + 1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False, decode=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse, decode=decode)  # the initial vertex
        for v_ in prop_order[1:]:
            # print(v_)
            self._propagate_to(G, v_, propagator, reverse=reverse, decode=decode)
            # Hv = self._propagate_to(G, v_, propagator, Hv, reverse=reverse) no need
        return Hv

    def _update_v(self, G, v, H0=None, decode=False):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False, decode=decode)
        return

    def _get_vertex_state(self, G, v, prior_edge=False):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden(prior_edge=prior_edge)
            else:
                hv = g.vs[v]['H_forward']
                if prior_edge:
                    pos_ = self._one_hot([g.vs[v]['pos']], self.max_pos)
                    hv = torch.cat([hv, pos_], 1)
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount() - 1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)  # a linear model
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False, decode=False)
        if self.bidir:
            self._propagate_from(G, self.max_n - 1, self.grue_backward,
                                 H0=self._get_zero_hidden(len(G)), reverse=True, decode=False)
        Hg = self._get_graph_state(G)
        # print(Hg.shape)

        dfs_ = []
        for g in G:
            df_ = [0] * (3 * self.max_pos)
            for v_ in range(len(g.vs)):
                pos_ = g.vs[v_]['pos']
                df_[pos_ * 3 + 0] = g.vs[v_]['r']
                df_[pos_ * 3 + 1] = g.vs[v_]['c']
                df_[pos_ * 3 + 2] = g.vs[v_]['gm']
            dfs_.append(df_)
        Hdf = torch.FloatTensor(dfs_).to(self.get_device())
        Hd = self.df_enc(Hdf)
        Hg = torch.cat([Hg, Hd], dim=1)  # concatenate the topology embedding and subg feature embedding

        mu, logvar = self.fc1(Hg), self.fc2(Hg)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))

    def decode(self, z, stochastic=True, node_type_dic=NODE_TYPE, subg_node=SUBG_NODE, subg_con=SUBG_CON,
               subg_indi=SUBG_INDI):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        pred_dfs = self.df_fc(H0)

        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['r'] = 0.0
            g.vs[0]['c'] = 0.0
            g.vs[0]['gm'] = 0.0
            g.vs[0]['pos'] = 0
        self._update_v(G, 0, H0, decode=True)  # only at the 'beginning', we need a hidden state H0
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                pos_scores = self.add_pos(Hg)
                # Implement attention mechanism here
                attn_scores = self.attention_linear(torch.cat((type_scores, pos_scores), dim=1))
                attn_weights = F.softmax(attn_scores, dim=1)
                attended_type_scores = torch.matmul(attn_weights, type_scores)
                attended_pos_scores = torch.matmul(attn_weights, pos_scores)

                if stochastic:
                    type_probs = F.softmax(attended_type_scores, dim=1).cpu().detach().numpy()
                    pos_probs = F.softmax(attended_pos_scores, dim=1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i])
                                 for i in range(len(G))]
                    new_pos = [np.random.choice(range(self.max_pos), p=pos_probs[i])
                               for i in range(len(G))]
                else:
                    new_types = torch.argmax(attended_type_scores, dim=1).flatten().tolist()
                    new_pos = torch.argmax(attended_pos_scores, dim=1).flatten().tolist()

            for j, g in enumerate(G):
                if not finished[j]:
                    g.add_vertex(type=new_types[j])
                    g.vs[idx]['pos'] = new_pos[j]
                    g.vs[idx]['r'] = pred_dfs[j, new_pos[j] * 3 + 0]
                    g.vs[idx]['c'] = pred_dfs[j, new_pos[j] * 3 + 1]
                    g.vs[idx]['gm'] = pred_dfs[j, new_pos[j] * 3 + 2]

            self._update_v(G, idx, decode=True)
            # decide connections
            edge_scores = []
            for vi in range(idx - 1, -1, -1):
                Hvi = self._get_vertex_state(G, vi, prior_edge=True)
                H = self._get_vertex_state(G, idx, prior_edge=True)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE:
                        # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0)
                                            if v.index != g.vcount() - 1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount() - 1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount() - 1)
                self._update_v(G, idx, decode=True)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory

        return G
