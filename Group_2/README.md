# From Invariance to Equivariance: Predicting Quantum Chemical Properties using Graph Neural Networks

## Group Members
- Adrija Dutta
- Jingdan Chen
- Eliza Asani
- Huihang Qiu

---

## Abstract
Molecules, as geometric graph entities with rich 3D coordinates, have properties invariant or equivariant under transformations like translation, rotation, reflection, and permutation. This study examines the effectiveness of invariant GNN-based models and data transformations for predicting quantum chemical properties. It highlights the importance of message passing and geometric information while discussing their limitations compared to true equivariant architectures.

---

## Code Citation
1. **Dive into Graphs (DIG) Library** (used for SchNet implementation):  
   - Documentation: [https://diveintographs.readthedocs.io/en/latest/](https://diveintographs.readthedocs.io/en/latest/)  
   - Publication: [DIG: A Turnkey Library for Diving into Graph Deep Learning Research](https://www.jmlr.org/papers/v22/21-0343.html)  
     - Liu, M., Luo, Y., Wang, L., Xie, Y., Yuan, H., Gui, S., ... & Ji, S. (2021).  
   - GitHub Repository: [https://github.com/divelab/DIG](https://github.com/divelab/DIG)

2. **Dataset** (processed version of QM9):  
   - Gasteiger, Johannes, Janek Groß, and Stephan Günnemann.  
   - [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123)  
   - GitHub Repository: [https://github.com/gasteigerjo/dimenet/tree/master/data](https://github.com/gasteigerjo/dimenet/tree/master/data)

3. **Modified Behler and Parrinello's Symmetry Function (from ANI-2X in TorchANI)**:  
   - Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg.  
   - [TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451)  
   - GitHub Repository: [https://github.com/aiqm/torchani](https://github.com/aiqm/torchani)

---

## Code Modules

### Directories and Files
- **`./Chkpoints/`**:  
  Final checkpoints (with lowest Validation MAE) for SchNet, SchNet (*w/o* MP), SphereNet, and Base Models.

- **`./dataset/`**:  
  Processed and ready-to-import dataset.

- **`./dig/`**:  
  Revised DIG implementation with the following updates:  
  1. `dig/threedgraph/dataset/rotation.py`: Dataset processing for rotation case-study data.  
  2. `dig/threedgraph/method/FCNN`: Baseline model implementation.

- **`./visualize_xyz/`**:  
  XYZ structures for visualization (compatible with GaussView 6).

- **`./Baseline_Model/`**:  
  Implementation of all baseline models.

- **`./SphereNet_project/`**:  
  SphereNet implementation:  
  1. Hyperparameter grid search.  
  2. Logs for search and training in TensorBoard format.

- **`./SchNet_project.ipynb`**:  
  SchNet hyperparameter search, training, and ablation studies.

- **`./test_rotation.ipynb`**:  
  Evaluation of SchNet, SchNet (*w/o* MP), SphereNet, and BaseM2 on rotation case studies.

- **`./analyze_data_distribution.ipynb`**:  
  Visualize QM9 dataset.

---

