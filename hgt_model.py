"""Heterogeneous graph transformer classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HeteroHGT(nn.Module):
    """Project typed inputs, propagate over relations, and classify companies."""

    def __init__(self, metadata, in_dims, hidden_dim=256, out_dim=2,
                 num_layers=2, dropout=0.6, heads=2, edge_drop_rate=0.1,
                 classifier_kind="linear", residual_init=-0.2006706955,
                 classifier_hidden_dims=(128, 32), company_project_width=None,
                 other_project_width=None, mix_normalization=False,
                 projection_dropout=0.0):
        super().__init__()
        if heads < 1 or hidden_dim % heads:
            raise ValueError("heads must be positive and divide hidden_dim")
        if not 0 <= edge_drop_rate < 1:
            raise ValueError("edge_drop_rate must be in [0, 1)")
        if not 0 <= projection_dropout < 1:
            raise ValueError("projection_dropout must be in [0, 1)")
        self.node_types = list(metadata[0])
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.heads = heads
        self.edge_drop_rate = edge_drop_rate
        self.mix_normalization = mix_normalization
        company_dim = in_dims["company"]
        other_dims = {in_dims[node_type] for node_type in self.node_types if node_type != "company"}
        if len(other_dims) != 1:
            raise ValueError(f"All non-company nodes must share one input dimension: {other_dims}")
        other_dim = other_dims.pop()
        def projection(input_dim, middle_dim):
            middle_dim = hidden_dim if middle_dim is None else middle_dim
            if middle_dim < 1:
                raise ValueError("Projection middle width must be positive")
            layers = [nn.Linear(input_dim, middle_dim), nn.ReLU()]
            if projection_dropout:
                layers.append(nn.Dropout(projection_dropout))
            layers.append(nn.Linear(middle_dim, hidden_dim))
            return nn.Sequential(*layers)

        self.company_project = projection(company_dim, company_project_width)
        self.other_project = projection(other_dim, other_project_width)
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(num_layers)
        ])
        self.residual_weight = nn.Parameter(torch.tensor(float(residual_init)))
        if classifier_kind == "linear":
            self.classifier = nn.Linear(hidden_dim, out_dim)
        elif classifier_kind == "mlp":
            widths = tuple(int(width) for width in classifier_hidden_dims)
            if not widths or any(width <= 0 for width in widths):
                raise ValueError("classifier_hidden_dims must contain positive widths")
            layers = []
            current_width = hidden_dim
            for width in widths:
                layers.extend((nn.Linear(current_width, width), nn.ReLU(), nn.Dropout(dropout)))
                current_width = width
            layers.append(nn.Linear(current_width, out_dim))
            self.classifier = nn.Sequential(*layers)
        elif classifier_kind == "shallow":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        elif classifier_kind == "deep":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout * 0.7),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout * 0.3),
                nn.Linear(hidden_dim // 4, out_dim),
            )
        else:
            raise ValueError(f"Unknown classifier_kind: {classifier_kind}")

    def forward(self, data, x_override=None, use_residual=True, return_branches=False):
        """Return logits for company nodes."""
        x_dict = {}
        for node_type in self.node_types:
            value = (x_override or {}).get(node_type, getattr(data[node_type], "x", None))
            if value is None:
                value = torch.zeros(data[node_type].num_nodes, self.hidden_dim,
                                    device=next(self.parameters()).device)
            projection = self.company_project if node_type == "company" else self.other_project
            x_dict[node_type] = projection(value)
        company_initial = x_dict["company"]
        for conv in self.convs:
            # Draw a fresh edge mask for this layer on every training forward pass.
            edge_index_dict = data.edge_index_dict
            if self.training and self.edge_drop_rate:
                edge_index_dict = {
                    edge_type: edge_index[:, torch.rand(edge_index.size(1), device=edge_index.device)
                                           >= self.edge_drop_rate]
                    for edge_type, edge_index in edge_index_dict.items()
                }
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(value), p=self.dropout,
                                     training=self.training)
                      for key, value in x_dict.items()}
        company_features = x_dict["company"]
        if use_residual:
            alpha = torch.sigmoid(self.residual_weight)
            graph_branch = (F.layer_norm(company_features, (self.hidden_dim,))
                            if self.mix_normalization else company_features)
            initial_branch = (F.layer_norm(company_initial, (self.hidden_dim,))
                              if self.mix_normalization else company_initial)
            company_features = alpha * graph_branch + (1.0 - alpha) * initial_branch
        logits = self.classifier(company_features)
        if return_branches:
            return logits, (company_initial, x_dict["company"], company_features)
        return logits

    def get_alpha(self):
        """Return the learned residual mixing coefficient."""
        return torch.sigmoid(self.residual_weight).item()
