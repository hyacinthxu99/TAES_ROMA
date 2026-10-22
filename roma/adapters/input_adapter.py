"""Graph-conditioned channel adapters."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .channel_attention import ChannelWiseAttention

D_ATTN = 16
D_CORE = 16
D_GNN_IN = 1
D_GNN_HIDDEN = 4
D_GNN_OUT = 4
D_GNN_LAYERS = 1

class CausalGNNEncoder(nn.Module):
    """Encode a weighted graph into a shared embedding."""

    def __init__(self, in_dim=D_GNN_IN, hidden_dim=D_GNN_HIDDEN, out_dim=D_GNN_OUT, num_layers=D_GNN_LAYERS, permute_prob=0.5, device=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = max(1, int(num_layers))
        self.conv1 = GCNConv(in_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim) if self.num_layers > 1 else None
        self.device = device
        self.readout = nn.Linear(self.hidden_dim, self.out_dim)
        self.permute_prob = permute_prob
        self._graph_cache = {}

    def forward(self, x, batch_size):
        current_device = next(self.parameters()).device
        (node_x, edge_index, edge_weight) = self.cached_gcn_tensors(x, current_device)
        node_x = F.relu(self.conv1(node_x, edge_index, edge_weight))
        if self.conv2 is not None:
            node_x = F.relu(self.conv2(node_x, edge_index, edge_weight))
        graph_embedding = node_x.mean(dim=0, keepdim=True)
        V_G = self.readout(graph_embedding)
        return V_G.expand(int(batch_size), -1)

    def cached_gcn_tensors(self, graph, device, threshold: float=1e-06):
        if isinstance(graph, torch.Tensor):
            graph_array = graph.detach().cpu().numpy()
            graph_id = id(graph)
        else:
            graph_array = np.asarray(graph)
            graph_id = id(graph)
        key = (graph_id, graph_array.shape, str(device), self.in_dim)
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached
        (n_rows, n_cols) = graph_array.shape
        if n_rows != n_cols:
            raise ValueError('graph must be a square matrix (N, N).')
        mask = np.abs(graph_array) > threshold
        np.fill_diagonal(mask, False)
        edge_pairs = np.argwhere(mask)
        if edge_pairs.size == 0:
            edge_pairs = np.arange(n_rows)[:, None].repeat(2, axis=1)
            edge_values = np.full(n_rows, 1e-06, dtype=np.float32)
        else:
            edge_values = graph_array[mask].astype(np.float32, copy=False)
        node_x = torch.ones((n_rows, self.in_dim), dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(edge_pairs.T, dtype=torch.long, device=device).contiguous()
        edge_weight = torch.as_tensor(edge_values, dtype=torch.float32, device=device)
        cached = (node_x, edge_index, edge_weight)
        self._graph_cache = {key: cached}
        return cached

class QueryHypernetwork(nn.Module):
    """Generate attention queries from the graph embedding."""

    def __init__(self, in_dim=D_GNN_OUT, out_flat_dim=D_CORE * D_ATTN):
        super().__init__()
        self.out_flat_dim = out_flat_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.ReLU(), nn.Linear(2 * in_dim, self.out_flat_dim))

    def forward(self, V_G):
        """ V_G: (Batch_size, D_GNN_OUT) """
        Q_flat = self.mlp(V_G)
        return Q_flat

class DynamicChannelAttentionLayer(nn.Module):

    def __init__(self, D_in, D_hidden, Seq_len, D_out, D_hyper_flat=D_CORE * D_ATTN, query_projection_mode='factorized'):
        super().__init__()
        self.D_in = D_in
        self.D_hidden = D_hidden
        self.D_out = D_out
        self.D_hyper_flat = D_hyper_flat
        self.Seq_len = Seq_len
        self.query_projection_mode = str(query_projection_mode).lower()
        self.key = nn.Linear(self.Seq_len, D_hidden)
        self.value = nn.Linear(self.Seq_len, self.Seq_len)
        if self.query_projection_mode == 'mlp':
            self.P_Q = nn.Sequential(nn.Linear(D_hyper_flat, 2 * D_hyper_flat), nn.ReLU(), nn.Linear(2 * D_hyper_flat, D_in * D_hidden))
        elif self.query_projection_mode == 'linear':
            self.P_Q = nn.Linear(D_hyper_flat, D_in * D_hidden)
        elif self.query_projection_mode == 'factorized':
            self.channel_query_basis = nn.Parameter(torch.empty(D_in, D_hidden))
            nn.init.xavier_uniform_(self.channel_query_basis)
            self.query_feature_gate = nn.Sequential(nn.Linear(D_hyper_flat, D_hidden), nn.Tanh())
            self.query_channel_gate = nn.Sequential(nn.Linear(D_hyper_flat, D_in), nn.Tanh())
        else:
            raise ValueError('query_projection_mode must be one of: factorized, linear, mlp')
        self.out_projection = nn.Linear(D_in, D_out)

    def project_query(self, Q_hyper_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.query_projection_mode == 'factorized':
            feature_gate = self.query_feature_gate(Q_hyper_flat).unsqueeze(1)
            channel_gate = self.query_channel_gate(Q_hyper_flat).unsqueeze(-1)
            query_basis = self.channel_query_basis.unsqueeze(0)
            return query_basis * (1.0 + feature_gate) * (1.0 + channel_gate)
        Q_pred_flat = self.P_Q(Q_hyper_flat)
        return Q_pred_flat.view(batch_size, self.D_in, self.D_hidden)

    def forward(self, X: torch.Tensor, Q_hyper: torch.Tensor):
        (B, T, D_in) = X.shape
        X_T = X.transpose(-2, -1)
        Q_hyper_flat = Q_hyper.view(B, -1)
        Q_pred = self.project_query(Q_hyper_flat, B)
        K = self.key(X_T)
        V = self.value(X_T)
        K_T = K.transpose(-2, -1)
        attn_scores = torch.matmul(Q_pred, K_T) / self.D_in ** 0.5
        attn_matrix = F.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_matrix, V)
        output = self.out_projection(attention_output.transpose(-2, -1))
        return (output, attn_matrix)

class InputAdapter(nn.Module):
    """Map target channels to the pretrained backbone width."""

    def __init__(self, input_dim, output_dim, causal_map=None, seq_len=30, d_attn=D_ATTN, d_core=D_CORE, gnn_in=D_GNN_IN, gnn_hidden=D_GNN_HIDDEN, gnn_out=D_GNN_OUT, gnn_layers=D_GNN_LAYERS, query_projection_mode='factorized', device=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_attn = d_attn
        self.d_core = d_core
        self.gnn_in = gnn_in
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out
        self.gnn_layers = gnn_layers
        self.query_projection_mode = query_projection_mode
        self.seq_len = seq_len
        self.device = device
        self.gnn = CausalGNNEncoder(in_dim=self.gnn_in, hidden_dim=self.gnn_hidden, out_dim=self.gnn_out, num_layers=self.gnn_layers, device=device)
        self.hyper = QueryHypernetwork(in_dim=self.gnn_out, out_flat_dim=self.d_core)
        self.attn_adapter = DynamicChannelAttentionLayer(D_in=self.input_dim, D_hidden=self.d_attn, Seq_len=self.seq_len, D_out=self.output_dim, D_hyper_flat=self.d_core, query_projection_mode=self.query_projection_mode)
        self.causal_map = causal_map

    def forward(self, x):
        V_G = self.gnn(self.causal_map, x.shape[0])
        Q_pred = self.hyper(V_G)
        (output, attn_matrix) = self.attn_adapter(x, Q_pred)
        return (output, attn_matrix)

class InputAdapterBase(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=30, d_attn=D_ATTN, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_attn = d_attn
        self.seq_len = seq_len
        self.device = device
        self.attn_adapter = ChannelWiseAttention(input_size=self.input_dim, output_size=output_dim, seq_length=self.seq_len, hidden_size=self.d_attn, predict_length=None)

    def forward(self, x):
        (output, attn_matrix) = self.attn_adapter(x)
        return (output, attn_matrix)

class InputAdapterLinear(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=30, d_attn=D_ATTN, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_attn = d_attn
        self.seq_len = seq_len
        self.device = device
        self.adapter = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        output = self.adapter(x)
        return (output, output)
