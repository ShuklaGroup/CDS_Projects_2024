import torch
import torch_geometric
import numpy as np
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SchNet, SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
from sklearn.model_selection import KFold
import os
import time

# Get the dataset
dataset = QM93D(root='dataset/')
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed =42)
train_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['test']] # mix the train and valid idxs

# Define hyperparameters
cutoffs = np.array([5.0, 10.0, 20.0])
num_sphericals = np.array([3,5,7])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Define 3-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Hyperparameter grid search with 3-fold CV
for cutoff in cutoffs:
    for num_spherical in num_sphericals:
        print(f"Hyperparameter set: cutoff={cutoff}, num_spherical={num_spherical}")

        cv_scores = []
        fold = 1
        for train_idx, valid_idx in kf.split(train_dataset):
            print(f"Fold {fold}...")

            t_start = time.time()
            train_data = train_dataset[train_idx]
            val_data = train_dataset[valid_idx]

            # build the model
            model = SphereNet(
                energy_and_force = False,
                cutoff = cutoff,
                num_layers = 6,
                hidden_channels = 128,
                out_channels = 1,
                int_emb_size = 64,
                basis_emb_size_dist = 8,
                basis_emb_size_angle=8, 
                basis_emb_size_torsion=8, 
                out_emb_channels=256,
                num_spherical=num_spherical, 
                num_radial=6, 
                envelope_exponent=5,
                num_before_skip=1, 
                num_after_skip=2, 
                num_output_layers=3
            ).to(device)

            # Early stopping and checkpointing
            save_path = f"./save_checkpoint/cutoff{cutoff}_spherical{num_spherical}_fold{fold}"
            log_path = f"./run/cutoff{cutoff}_spherical{num_spherical}_fold{fold}"

            # Training loop
            run3d = run()
            run3d.run(
                device=device,
                train_dataset=train_data,
                valid_dataset=val_data,
                test_dataset=test_dataset,
                model=model,
                loss_func=loss_func,
                evaluation=evaluation,
                epochs=120,
                batch_size=256,
                vt_batch_size=256,
                lr=0.0005,
                lr_decay_factor=0.5,
                lr_decay_step_size=15,
                save_dir=save_path,
                log_dir=log_path,
            )

            t_end = time.time()
            # minutes
            t_spend = (t_end - t_start) / 60
            print(f"Time: {t_spend} min")

            torch.cuda.empty_cache()
            fold+=1