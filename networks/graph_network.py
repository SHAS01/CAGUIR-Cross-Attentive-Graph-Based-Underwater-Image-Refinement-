# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph neural network imports
from torch_geometric.nn import GCNConv

# Custom imports
from networks.network import net  # Import the base net class
from graphmethods import dense_to_sparse
from networks.blocks import DAM  # Import DAM module
from networks.enhance_net_nopool_v2 import enhance_net_nopool  # Import the new enhance_net_nopool

##### Depth-guided enhancement network  #####
##### input:256*256*3, 256*256*1|output:256*256*3

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class graph_net(nn.Module):
    def __init__(self, block_size=16, in_c=3, out_c=3, hid_size=1024, bc=64, hidden_size=1500, num_layers=2):
        super(graph_net, self).__init__()
        self.hid_size = hid_size

        # Base net
        self.base_net = net(in_c, out_c, bc)

        # Attention modules
        self.dam_down1 = DAM(bc)
        self.dam_down2 = DAM(bc)
        self.dam_gnn = DAM(hid_size)
        self.dam_up = DAM(bc)

        # Projection layer
        self.feature_proj = nn.Conv2d(bc, hid_size, kernel_size=1)

        # GNN Layer
        self.GCN = GCN(hid_size, hidden_size, hid_size)

        # Transfer block
        self.transfer = nn.Conv2d(hid_size, bc, kernel_size=1)

        # Enhance net (new enhance_net_nopool)
        self.enhance_net = enhance_net_nopool(scale_factor=1)  # Adjust scale_factor as needed

    def forward(self, x, adj):
        # ---- Encoder ----
        e1 = self.base_net.conv_ini(x)
        e1 = self.base_net.rb1(e1)
        e2 = self.base_net.down_conv1(e1)
        e2 = self.dam_down1(e2, e2)  # Pass e2 as both x1 and x2
        e2 = self.base_net.mrb1(e2)
        e3 = self.base_net.down_conv2(e2)
        e3 = self.dam_down2(e3, e3)  # Pass e3 as both x1 and x2
        e3 = self.base_net.mrb2(e3)

        # ---- GNN Processing ----
        projected = self.feature_proj(e3)
        projected = self.dam_gnn(projected, projected)  # Pass projected as both x1 and x2
        B, _, H, W = projected.shape
        proj_reshaped = projected.permute(0, 2, 3, 1).reshape(-1, self.hid_size)
        sparse_adj = dense_to_sparse(adj).to(x.device)
        gcn_out = self.GCN(proj_reshaped, sparse_adj)
        gcn_out = gcn_out.view(B, H, W, self.hid_size).permute(0, 3, 1, 2)
        e3_gcn = gcn_out + projected
        e3_transferred = self.transfer(e3_gcn)

        # ---- Decoder ----
        d3 = self.base_net.up_conv1(e3_transferred) + e2
        d3 = self.dam_up(d3, d3)  # Pass d3 as both x1 and x2
        d3 = self.base_net.mrb3(d3)
        d2 = self.base_net.up_conv2(d3) + e1
        d2 = self.base_net.rb2(d2)
        out = self.base_net.conv_last(d2)

        # ---- Enhance Net Integration ----
        enhance_image, _ = self.enhance_net(out)

        return enhance_image