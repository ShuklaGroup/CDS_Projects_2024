# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/combined_molecule_data_with_alkene_type.xlsx'
# df = pd.read_excel(file_path)

# # Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
# labels = df['Molecule']
# alkene_types = df['_Alkene_Type']

# # Drop irrelevant features and non-numeric columns, but keep '_Alkene_Type'
# # df = df.drop(columns=['Rxn ID', 'Pressure/atm', 'Temperature/C', 'S/C', 'ddG'])

# # Select only numeric columns
# df_numeric = df.select_dtypes(include=[float, int])

# # Drop rows with any missing values
# df_numeric = df_numeric.dropna()

# # Ensure 'alkene_types' aligns with the numeric dataframe
# alkene_types = alkene_types[df_numeric.index]

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Fit the scaler on the data
# scaler.fit(df_numeric)

# # Transform the data using the fitted scaler
# df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)

# # Apply VarianceThreshold to remove low-variance features
# selector = VarianceThreshold(threshold=0)
# df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # Apply PCA to Reactant Features
# reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
# pca_reactants = PCA(n_components=2)
# reactant_pca = pca_reactants.fit_transform(df_selected[reactant_features])

# # Convert the PCA-transformed features back to a DataFrame
# reactant_pca_df = pd.DataFrame(reactant_pca, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca.shape[1])])

# # Visualization of PCA for Reactant Features colored by _Alkene_Type
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=reactant_pca_df['Reactant_PCA1'], y=reactant_pca_df['Reactant_PCA2'], hue=alkene_types, palette='viridis', s=100)

# plt.title('PCA of Reactant Features Colored by _Alkene_Type')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title='_Alkene_Type')
# plt.grid(True)
# plt.show()

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D

# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/combined_molecule_data_with_alkene_type.xlsx'
# df = pd.read_excel(file_path)

# # Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
# labels = df['Molecule']
# alkene_types = df['_Alkene_Type']

# # Drop irrelevant features and non-numeric columns, but keep '_Alkene_Type'
# # Select only numeric columns
# df_numeric = df.select_dtypes(include=[float, int])

# # Drop rows with any missing values
# df_numeric = df_numeric.dropna()

# # Ensure 'alkene_types' aligns with the numeric dataframe
# alkene_types = alkene_types[df_numeric.index]

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Fit the scaler on the data
# scaler.fit(df_numeric)

# # Transform the data using the fitted scaler
# df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)

# # Apply VarianceThreshold to remove low-variance features
# selector = VarianceThreshold(threshold=0)
# df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # Apply PCA to Reactant Features with 3 components
# reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
# pca_reactants = PCA(n_components=3)
# reactant_pca = pca_reactants.fit_transform(df_selected[reactant_features])

# # Convert the PCA-transformed features back to a DataFrame
# reactant_pca_df = pd.DataFrame(reactant_pca, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca.shape[1])])

# # Visualization of PCA for Reactant Features colored by _Alkene_Type in 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(reactant_pca_df['Reactant_PCA1'], 
#                 reactant_pca_df['Reactant_PCA2'], 
#                 reactant_pca_df['Reactant_PCA3'], 
#                 c=alkene_types.map({label: idx for idx, label in enumerate(alkene_types.unique())}), 
#                 cmap='viridis', s=10)

# ax.set_title('PCA of Reactant Features Colored by _Alkene_Type')
# ax.set_xlabel('PCA Component 1')
# ax.set_ylabel('PCA Component 2')
# ax.set_zlabel('PCA Component 3')

# # Adding legend manually
# # Mapping unique alkene types to the colors
# unique_alkene_types = alkene_types.unique()
# handles, _ = sc.legend_elements(prop="colors", alpha=0.6)
# legend_labels = [unique_alkene_types[int(h.get_label())] for h in handles if h.get_label().isdigit()]
# ax.legend(handles, legend_labels, title="_Alkene_Type")

# plt.show()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/combined_molecule_data_with_alkene_type.xlsx'
df = pd.read_excel(file_path)

# Step 1: Filter the DataFrame for Di-substituted alkenes
di_sub_df = df[df['_Alkene_Type'] == 'Di-sub']

# Preserve the 'Molecule' and '_Alkene_Type' for labeling
labels = di_sub_df['Molecule']
alkene_types = di_sub_df['_Alkene_Type']

# Step 2: Select only numeric columns
df_numeric = di_sub_df.select_dtypes(include=[float, int])

# Drop rows with any missing values
df_numeric = df_numeric.dropna()

# Step 3: Apply MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# Step 4: Apply VarianceThreshold to remove low-variance features
selector = VarianceThreshold(threshold=0)
df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# Step 5: Define 'reactant_features' and exclude Q2 and Q3 (i.e., throw out features related to hydrogens)
# Assuming columns like 'Q2_B1', 'Q3_B1', etc. exist
reactant_features = [col for col in df_selected.columns if 'Q2' not in col and 'Q3' not in col and ('Q' in col or 'C' in col)]

# reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
# Step 6: Apply PCA for 3D visualization
pca_reactants_3d = PCA(n_components=3)
reactant_pca_3d = pca_reactants_3d.fit_transform(df_selected[reactant_features])

# Convert the PCA-transformed features back to a DataFrame
reactant_pca_3d_df = pd.DataFrame(reactant_pca_3d, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca_3d.shape[1])])

# Step 7: 3D Visualization of PCA for Reactant Features colored by 'Molecule'
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(reactant_pca_3d_df['Reactant_PCA1'], 
                reactant_pca_3d_df['Reactant_PCA2'], 
                reactant_pca_3d_df['Reactant_PCA3'], 
                c=labels.map({label: idx for idx, label in enumerate(labels.unique())}), 
                cmap='Set2', s=100)

# Add axis labels and title
ax.set_title('PCA of Di-sub Alkenes (3D)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

# Add legend
unique_labels = labels.unique()
handles, _ = sc.legend_elements(prop="colors", alpha=0.6)
legend_labels = [unique_labels[int(h.get_label())] for h in handles if h.get_label().isdigit()]
ax.legend(handles, legend_labels, title="Molecule")

plt.show()
