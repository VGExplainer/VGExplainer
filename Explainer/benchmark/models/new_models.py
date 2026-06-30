
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv,GCNConv,global_mean_pool,global_max_pool, global_add_pool,TopKPooling,GlobalAttention





class DevignModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_steps=6, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(out_channels=self.hidden_dim, num_layers=self.num_timesteps)
        self.concat_dim = self.input_dim + self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.concat_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        node_embeddings = self.ggnn(x, edge_index)
        concatenated_features = torch.cat((x, node_embeddings), dim=1)
        graph_embedding = global_mean_pool(concatenated_features, batch)
        logits = self.mlp(graph_embedding)
        return logits

class IVDetect(nn.Module):

    def __init__(self, output_dim=256, input_dim=128, num_classes=2):
        super().__init__()
        self.out_dim = output_dim
        self.in_dim = input_dim
        self.conv1 = GCNConv(input_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        self.conv3 = GCNConv(output_dim, num_classes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)##0.3
        self.connect = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch):
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv,edge_index))
        post_conv = self.conv3(post_conv,edge_index)
        pooled = global_max_pool(post_conv, batch)
        return pooled

class RevealModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2, num_steps=6, dropout_rate=0.2, **kwargs):
        """
        構造函數
        Args:
            input_dim (int): 輸入節點特徵的維度。
            hidden_dim (int): GNN 隱藏層的維度。這也將作為後續 MLP 的基礎維度。
            num_classes (int): 最終分類的類別數。
            gnn_steps (int): GatedGraphConv 的時間步/層數。
            dropout_rate (float): Dropout 的比率。
            **kwargs: 用於接收來自 Pipeline 但此模型可能用不到的參數，以增強兼容性。
        """
        super().__init__()

        # --- GNN 部分 ---
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_steps)
        # mlp_hidden_dim =  int(hidden_dim/2)
        mlp_hidden_dim =  hidden_dim*2

        self.feature_extractor_mlp = nn.Sequential(
            # Layer 1: 從 GNN 輸出的維度擴展到 MLP 隱藏維度
            nn.Linear(in_features=hidden_dim, out_features=mlp_hidden_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # Layer 2: 模擬原始 ExtractFeature 的內部結構
            nn.Linear(in_features=mlp_hidden_dim, out_features=hidden_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # Layer 3: 再次擴展回 MLP 隱藏維度
            nn.Linear(in_features=mlp_hidden_dim, out_features=hidden_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)


    def forward(self, x, edge_index, batch):
        # 1. GNN 提取節點嵌入
        node_embeddings = self.ggnn(x, edge_index)
        
        # 2. 池化層將節點嵌入聚合為圖嵌入
        #    原始模型使用 add pool，我們在此遵循
        graph_embedding = global_add_pool(node_embeddings, batch)
        
        # 3. 使用 MLP 進一步提取圖級別的特徵
        graph_features = self.feature_extractor_mlp(graph_embedding)
        
        # 4. 使用分類器得到最終的 logits
        logits = self.classifier(graph_features)
        
        return logits

class DeepWukongModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_classes=2, dropout_rate=0.5, top_k_ratio=0.8, **kwargs):
        """
        構造函數
        Args:
            input_dim (int): 輸入節點特徵的維度。
            hidden_dim (int): GCN 和注意力池化層的隱藏維度。
            num_classes (int): 最終分類的類別數。
            dropout_rate (float): MLP 中 Dropout 的比率。
            top_k_ratio (float): TopKPooling 層保留節點的比例。
            **kwargs: 用於接收來自 Pipeline 但此模型可能用不到的參數，以增強兼容性。
        """
        super().__init__()

        # --- 圖編碼器部分 (原 GraphConvEncoder 的邏輯) ---
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.pooling = TopKPooling(hidden_dim, ratio=top_k_ratio)
        # GlobalAttention 的門控網絡 gate_nn 需要一個 nn.Module
        self.attention_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        # --- MLP 分類器部分 (原 DeepWukong 的頂層邏輯) ---
        # 遵循原始設計，MLP的隱藏維度是GNN隱藏維度的兩倍
        mlp_hidden_dim = 2 * hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.classifier = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        """
        前向傳播函數
        Args:
            x (Tensor): 節點特徵 [num_total_nodes, input_dim]
            edge_index (LongTensor): 邊索引 [2, num_total_edges]
            batch (LongTensor): 批次向量 [num_total_nodes]，將每個節點映射到其所在的圖。
        
        Returns:
            Tensor: 每個圖的分類 logits [batch_size, num_classes]
        """
        # 1. GCN 層提取初步的節點特徵
        node_embedding = self.gcn_conv(x, edge_index)
        node_embedding = F.relu(node_embedding)

        # 2. TopKPooling 層對節點進行篩選
        #    關鍵修正：必須傳入 `batch` 參數，並捕獲更新後的 `x`, `edge_index`, `batch`
        node_embedding, edge_index, _, batch, _, _ = self.pooling(
            x=node_embedding,
            edge_index=edge_index,
            batch=batch
        )

        # 3. GlobalAttention 層對篩選後的節點進行加權池化，得到圖級別的表示
        #    關鍵修正：必須使用 TopKPooling 返回的更新後的 `batch` 向量
        graph_embedding = self.attention_pool(
            x=node_embedding,
            batch=batch
        )
        
        # 4. 將圖級別表示送入 MLP 進行深度特徵提取
        hidden_features = self.mlp(graph_embedding)
        
        # 5. 使用最終的分類器得到 logits
        logits = self.classifier(hidden_features)
        
        return logits