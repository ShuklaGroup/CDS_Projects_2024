# CDS-Group9-Project
This is the group project of Group 9 for the CDS Course at UIUC - Fall 2024.

## Title
**Property-Driven Molecule Generation using Conditional Normalizing Flows**

## Authors
Omkar Chaudhari, Akshay Gurumoorthi, Rishabh Puri, Matthew Too

## Abstract
This work presents a method to generate molecules with target properties using Conditional Normalising Flows. The properties of the generated molecules are further validated using xtb and ORCA. Using the QM9 dataset, an autoregressive normalising flow model is trained on the molecules using TensorFlow and DeepChem, with the one-hot encodings of their SELFIES strings. The model is conditioned during training with DFT-computed properties of the molecules from the dataset. By utilising the learned bijections, new molecules are sampled by passing a condition vector with properties of interest. Using the model's outputted SELFIES strings, these are converted back to SMILES, created into ORCA input files using RDKit and Python, and validated using ORCA. This conditioning approach is tested to see if the model can be directed toward generating molecules with desired properties. Such models will be useful to generate new molecules in various domains like drug discovery, the semiconductor industry, renewable materials, etc.

## Code Citation
The base code was used from DeepChem’s normalising flow tutorial: [DeepChem Normalizing Flow Tutorial](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Training_a_Normalizing_Flow_on_QM9.ipynb).

The initial SMILES processing is the same as the tutorial, but the model implementation has been changed and written by us. The conditioning part had not been implemented into the DeepChem library’s normalising flow function.

Additionally, we have added more parameter controls, early stopping, and have added the DFT conditioning sampling from the model. The `nfm.flow.sample()` function by DeepChem cannot sample conditioned samples, so we used TensorFlow’s `tfd.TransformedDistribution.sample` to sample the conditioned molecule from the learned chained bijections.

## Hyperparameter Tuning
Hyperparameter tuning was performed using 6 separate notebooks to take advantage of parallel sessions on Kaggle. The results of the tuning are stored in the format `{layers}_{hidden units}`. Each notebook explored a different combination of layers and hidden units to optimize model performance efficiently.

## Validation and Error Analysis
- A `validation.ipynb` notebook is provided, which computes the properties of the generated molecules using ORCA.
- An `error.ipynb` notebook compares the mean absolute error of a specified property across the different hyperparameter combinations.

## Results
- Generated molecules exhibit the desired properties as validated using xtb and ORCA.
- The model demonstrates the ability to generate molecules for applications in drug discovery, semiconductor materials, and renewable energy.

---

For additional details, refer to the project notebooks and results.
