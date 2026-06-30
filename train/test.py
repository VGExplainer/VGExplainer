import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Import Necessary Libraries ---
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool, TopKPooling, GlobalAttention

# --- 1. Configuration Class ---
class Config:
    DATA_DIR = "/home/Dataset/primevul/4_embedding/pdg_norm"
    BEST_MODEL_PATH = "/home/VGExplainer/train_model/primevul/model/deepwukong/"
    SPLIT_FILE_PATH = "/home/VGExplainer/train_model/primevul/dataset/ori_split_files_norm.json"
    DIFF_DIR = "/home/Dataset/primevul/0_src/diff"  # Directory containing patches (.diff files)
    INPUT_DIM = 128
    HIDDEN_DIM = 256
    NUM_GGNN_STEPS = 4
    NUM_CLASSES = 2
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_SPLIT_RATIO = 0.15
    VALIDATION_SPLIT_RATIO = 0.15

class DevignModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, num_classes):
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
    def __init__(self, output_dim, input_dim, num_classes):
        super().__init__()
        self.out_dim = output_dim # 200
        self.in_dim = input_dim
        self.conv1 = GCNConv(input_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        self.conv3 = GCNConv(output_dim, num_classes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.connect = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch):
        post_conv = self.relu1(self.conv1(x, edge_index))
        post_conv = self.dropout(post_conv)
        post_conv = self.connect(post_conv)
        post_conv = self.relu2(self.conv2(post_conv, edge_index))
        post_conv = self.conv3(post_conv, edge_index)
        pooled = global_max_pool(post_conv, batch)
        return pooled

class RevealModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, dropout_rate=0.2):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_steps)
        
        # Simulates internal scaling logic from original code
        internal_dim = int(hidden_dim / 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(internal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification layer with LogSoftmax
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, edge_index, batch):
        node_embeddings = self.ggnn(x, edge_index)
        # Original model uses global_add_pool
        graph_embedding = global_add_pool(node_embeddings, batch)
        
        # Deep feature extraction
        features = self.feature_extractor(graph_embedding)
        logits = self.classifier(features)
        
        return logits

class DeepWukongModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5, top_k_ratio=0.8, **kwargs):
        """
        Constructor
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Hidden dimension for GCN and attention pooling layers.
            num_classes (int): Number of target classes.
            dropout_rate (float): Dropout ratio in MLP.
            top_k_ratio (float): Ratio of nodes retained by TopKPooling.
            **kwargs: Extra arguments for pipeline compatibility.
        """
        super().__init__()

        # --- Graph Encoder (GraphConvEncoder logic) ---
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.pooling = TopKPooling(hidden_dim, ratio=top_k_ratio)
        # Gating network for GlobalAttention
        self.attention_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        # --- MLP Classifier (DeepWukong top-level logic) ---
        # MLP hidden dimension is twice the GNN hidden dimension
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
        Forward Pass
        Args:
            x (Tensor): Node features [num_total_nodes, input_dim]
            edge_index (LongTensor): Edge indices [2, num_total_edges]
            batch (LongTensor): Batch vector [num_total_nodes]
        
        Returns:
            Tensor: Logits for each graph [batch_size, num_classes]
        """
        # 1. Feature extraction with GCN
        node_embedding = self.gcn_conv(x, edge_index)
        node_embedding = F.relu(node_embedding)

        # 2. Node pruning with TopKPooling
        # Key Fix: Must pass 'batch' and catch updated x, edge_index, and batch
        node_embedding, edge_index, _, batch, _, _ = self.pooling(
            x=node_embedding,
            edge_index=edge_index,
            batch=batch
        )

        # 3. Graph-level representation via GlobalAttention pooling
        # Key Fix: Use the updated batch vector from TopKPooling
        graph_embedding = self.attention_pool(
            x=node_embedding,
            batch=batch
        )
        
        # 4. Deep feature extraction via MLP
        hidden_features = self.mlp(graph_embedding)
        
        # 5. Final classification
        logits = self.classifier(hidden_features)
        
        return logits

# --- 3. Data Loading ---
def load_graph_from_json(filepath, flip_list):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    basename = os.path.basename(filepath)
    if flip_list:
        if basename in flip_list:
            original_label = data_dict['target']
            data_dict['target'] = 1 - original_label
    if not data_dict.get('node_features'):
        return None
    x = torch.tensor(data_dict['node_features'], dtype=torch.float32)
    edge_index_list = [[src, dst] for src, _, dst in data_dict['graph'] if src < len(x) and dst < len(x)]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([data_dict['target']], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, name=basename)

def load_dataset_from_filenames(data_dir, filenames, flip_list=None):
    dataset = []
    for fname in tqdm(filenames, desc="Loading data", leave=False):
        filepath = os.path.join(data_dir, fname)
        graph_data = load_graph_from_json(filepath, flip_list)
        if graph_data:
            dataset.append(graph_data)
    return dataset

def load_and_manage_splits_by_filename(data_dir, split_file_path, flip_list):
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as f:
            split_files = json.load(f)
        val_dataset = load_dataset_from_filenames(data_dir, split_files['val'], flip_list)
        test_dataset = load_dataset_from_filenames(data_dir, split_files['test'], flip_list)
        return val_dataset, test_dataset
    # Logics for re-splitting can be added here if needed
    return None, None

# --- 4. Evaluation Function ---
def evaluate(model, loader, criterion, device):
    """
    Evaluates model and returns labels, predictions, and filenames.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_names = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # batch.name is packed into a list in DataLoader
            all_names.extend(batch.name)
    
    avg_loss = total_loss / len(loader)
    return avg_loss, all_labels, all_preds, all_names

# --- 5. Logic to Save Correct Samples ---
def process_and_save_correct_samples(labels, preds, names, diff_dir, output_prefix):
    correct_0 = []
    correct_1 = []

    for label, pred, name in zip(labels, preds, names):
        if label == pred:
            # Check if corresponding .diff file exists (replace .json with .diff)
            patch_name = name[2:].replace('.json', '.diff')
            patch_path = os.path.join(diff_dir, patch_name)
            has_patch = os.path.exists(patch_path)
            
            entry = name
            if has_patch:
                if label == 0:
                    correct_0.append(entry)
                else:
                    correct_1.append(entry)

    with open(f"{Config.BEST_MODEL_PATH}{output_prefix}_correct_class_0.txt", "w") as f:
        f.write("\n".join(correct_0))
    with open(f"{Config.BEST_MODEL_PATH}{output_prefix}_correct_class_1.txt", "w") as f:
        f.write("\n".join(correct_1))
    
    print(f"[{output_prefix}] Correct samples saved: Class 0: {len(correct_0)}, Class 1: {len(correct_1)}")

# --- 6. Main Process ---
if __name__ == '__main__':
    flip_list = None
    val_data, test_data = load_and_manage_splits_by_filename(Config.DATA_DIR, Config.SPLIT_FILE_PATH, flip_list)

    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = DeepWukongModel(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Load existing model
    best_model_filename = '/home/VGExplainer/train_model/primevul/deepwukong/f1_0.5771_acc_0.5567_pre_0.5522_rec_0.6044_best.pt'
    
    if os.path.exists(best_model_filename):
        print(f"Loading model: {os.path.basename(best_model_filename)}")
        model.load_state_dict(torch.load(best_model_filename))
        
        # 1. Evaluate and save validation set results
        print("\nEvaluating validation set...")
        _, val_labels, val_preds, val_names = evaluate(model, val_loader, criterion, Config.DEVICE)
        process_and_save_correct_samples(val_labels, val_preds, val_names, Config.DIFF_DIR, "validation")
        report = classification_report(val_labels, val_preds, digits=4)
        print(f"\nFinal Validation Classification Report:\n{report}")

        # 2. Evaluate and save test set results
        print("\nEvaluating test set...")
        test_loss, test_labels, test_preds, test_names = evaluate(model, test_loader, criterion, Config.DEVICE)
        process_and_save_correct_samples(test_labels, test_preds, test_names, Config.DIFF_DIR, "test")
        
        # Print test report
        report = classification_report(test_labels, test_preds, digits=4)
        print(f"\nFinal Test Classification Report:\n{report}")
    else:
        print(f"Model file does not exist: {best_model_filename}")