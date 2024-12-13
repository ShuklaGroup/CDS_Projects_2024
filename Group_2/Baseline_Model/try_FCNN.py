import os, sys
# sys.path.append('..')

from dig.threedgraph.dataset import QM93D
import torch
import torchani
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from dig.threedgraph.method import FCNN
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


dataset = QM93D(root='dataset/')
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
# split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=11000, valid_size=1000, seed=42)
# split_idx['test'] = split_idx['valid']
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]


model = FCNN(device=device)

# # Directory containing .pt files
# args_dir = "save_dir/valid_checkpoint10.pt"

# state_dict = torch.load(args_dir)['model_state_dict']
# model.load_state_dict(state_dict)


loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=50, batch_size=32, vt_batch_size=32, lr=0.001, lr_decay_factor=0.95, lr_decay_step_size=1,
          energy_and_force=False, p=None, save_dir='save_dir', log_dir='log_dir')