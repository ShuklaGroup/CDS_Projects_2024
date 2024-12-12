# import pandas as pd
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/reordered_final_merged_cleaned_qqm_molecule_charges.xlsx'
# df = pd.read_excel(file_path)

# # Preserve the 'Reaction ID' for labeling
# labels = df['Rxn ID']

# # Drop the irrelevant features
# df = df.drop(columns=['Rxn ID', 'Pressure/atm', 'Temperature/C', 'S/C', 'ddG', '_Alkene_Type', ])

# # Handle missing values by imputing with the mean
# imputer = SimpleImputer(strategy='mean')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Fit the scaler on the data
# scaler.fit(df_imputed)

# # Transform the data using the fitted scaler
# df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

# # Apply VarianceThreshold to remove low-variance features
# selector = VarianceThreshold(threshold=0)
# df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # Apply PCA to Reactant Features
# reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
# pca_reactants = PCA(n_components=2)
# reactant_pca = pca_reactants.fit_transform(df_selected[reactant_features])

# # Apply PCA to Catalyst Features
# catalyst_features = [col for col in df_selected.columns if 'Cat' in col]
# pca_catalysts = PCA(n_components=2)
# catalyst_pca = pca_catalysts.fit_transform(df_selected[catalyst_features])

# # Convert the PCA-transformed features back to DataFrames
# reactant_pca_df = pd.DataFrame(reactant_pca, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca.shape[1])])
# catalyst_pca_df = pd.DataFrame(catalyst_pca, columns=[f'Catalyst_PCA{i+1}' for i in range(catalyst_pca.shape[1])])

# # Combine the PCA-transformed features into a final DataFrame
# df_final_pca = pd.concat([reactant_pca_df, catalyst_pca_df], axis=1)

# # Save the resulting DataFrame to a new Excel file
# pca_output_path = 'pca_transformed_data.xlsx'
# df_final_pca.to_excel(pca_output_path, index=False)

# # Visualization of PCA for Reactant Features with labels
# plt.figure(figsize=(10, 7))
# plt.scatter(reactant_pca_df['Reactant_PCA1'], reactant_pca_df['Reactant_PCA2'], color='blue')

# # # Annotate each point with the corresponding label (Reaction ID)
# # for i, label in enumerate(labels):
# #     plt.annotate(label, (reactant_pca_df['Reactant_PCA1'][i], reactant_pca_df['Reactant_PCA2'][i]), fontsize=9, alpha=0.7)

# plt.title('PCA of Reactant Features')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.grid(True)
# plt.show()

# # Visualization of PCA for Catalyst Features with labels
# plt.figure(figsize=(10, 7))
# plt.scatter(catalyst_pca_df['Catalyst_PCA1'], catalyst_pca_df['Catalyst_PCA2'], color='green')

# # # Annotate each point with the corresponding label (Reaction ID)
# # for i, label in enumerate(labels):
# #     plt.annotate(label, (catalyst_pca_df['Catalyst_PCA1'][i], catalyst_pca_df['Catalyst_PCA2'][i]), fontsize=9, alpha=0.7)

# plt.title('PCA of Catalyst Features')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.grid(True)
# plt.show()

# print(f"PCA applied after MinMax scaling, handling NaN values, and removing low-variance features. Transformed data saved to {pca_output_path}.")




# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/formatted_reaction_data_with_ids.xlsx'
# df = pd.read_excel(file_path)

# # Drop the irrelevant features
# df = df.drop(columns=['Reaction ID', 'Pressure', 'Temperature', 'S/C', 'ddG', '_Alkene_Type'])

# # Handle missing values by imputing with the mean
# imputer = SimpleImputer(strategy='mean')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Fit the scaler on the data
# scaler.fit(df_imputed)

# # Transform the data using the fitted scaler
# df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

# # Apply VarianceThreshold to remove low-variance features
# selector = VarianceThreshold(threshold=0)
# df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # Apply PCA to the selected features (all features)
# pca = PCA(n_components=None)
# pca.fit(df_selected)

# # Explained variance ratio
# explained_variance_ratio = pca.explained_variance_ratio_

# # Cumulative explained variance
# cumulative_explained_variance = explained_variance_ratio.cumsum()

# # Plotting the explained variance
# plt.figure(figsize=(10, 7))
# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
# plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
# plt.xlabel('Principal Component Index')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by Principal Components')
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()






# '''
# plot PCA by _Alkene_Type
# '''
# # import pandas as pd
# # from sklearn.feature_selection import VarianceThreshold
# # from sklearn.decomposition import PCA
# # from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import MinMaxScaler
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Load the dataset
# # file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/formatted_reaction_data_with_ids.xlsx'
# # df = pd.read_excel(file_path)

# # # Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
# # labels = df['Reaction ID']
# # alkene_types = df['_Alkene_Type']

# # # Drop the irrelevant features
# # df = df.drop(columns=['Reaction ID', 'Pressure', 'Temperature', 'S/C', 'ddG', '_Alkene_Type'])

# # # Handle missing values by imputing with the mean
# # imputer = SimpleImputer(strategy='mean')
# # df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # # Initialize the MinMaxScaler
# # scaler = MinMaxScaler()

# # # Fit the scaler on the data
# # scaler.fit(df_imputed)

# # # Transform the data using the fitted scaler
# # df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

# # # Apply VarianceThreshold to remove low-variance features
# # selector = VarianceThreshold(threshold=0)
# # df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # # Apply PCA to Reactant Features
# # reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
# # pca_reactants = PCA(n_components=2)
# # reactant_pca = pca_reactants.fit_transform(df_selected[reactant_features])

# # # Apply PCA to Catalyst Features
# # catalyst_features = [col for col in df_selected.columns if 'Cat' in col]
# # pca_catalysts = PCA(n_components=2)
# # catalyst_pca = pca_catalysts.fit_transform(df_selected[catalyst_features])

# # # Convert the PCA-transformed features back to DataFrames
# # reactant_pca_df = pd.DataFrame(reactant_pca, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca.shape[1])])
# # catalyst_pca_df = pd.DataFrame(catalyst_pca, columns=[f'Catalyst_PCA{i+1}' for i in range(catalyst_pca.shape[1])])

# # # Combine the PCA-transformed features into a final DataFrame
# # df_final_pca = pd.concat([reactant_pca_df, catalyst_pca_df], axis=1)

# # # Save the resulting DataFrame to a new Excel file
# # pca_output_path = 'pca_transformed_data.xlsx'
# # df_final_pca.to_excel(pca_output_path, index=False)

# # # Visualization of PCA for Reactant Features, colored by _Alkene_Type
# # plt.figure(figsize=(10, 7))
# # sns.scatterplot(x=reactant_pca_df['Reactant_PCA1'], y=reactant_pca_df['Reactant_PCA2'], hue=alkene_types, palette='viridis', s=100)

# # plt.title('PCA of Reactant Features')
# # plt.xlabel('PCA Component 1')
# # plt.ylabel('PCA Component 2')
# # plt.legend(title='_Alkene_Type')
# # plt.grid(True)
# # plt.show()

# # # Visualization of PCA for Catalyst Features, colored by _Alkene_Type
# # plt.figure(figsize=(10, 7))
# # sns.scatterplot(x=catalyst_pca_df['Catalyst_PCA1'], y=catalyst_pca_df['Catalyst_PCA2'], hue=alkene_types, palette='viridis', s=100)

# # plt.title('PCA of Catalyst Features')
# # plt.xlabel('PCA Component 1')
# # plt.ylabel('PCA Component 2')
# # # plt.legend(title='_Alkene_Type')
# # plt.grid(True)
# # plt.show()

# # print(f"PCA applied after MinMax scaling, handling NaN values, and removing low-variance features. Transformed data saved to {pca_output_path}.")

# ''''
# apply clusters 
# '''
# import pandas as pd
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/formatted_reaction_data_with_ids.xlsx'
# df = pd.read_excel(file_path)

# # Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
# labels = df['Reaction ID']
# alkene_types = df['_Alkene_Type']

# # Drop the irrelevant features
# df = df.drop(columns=['Reaction ID', 'Pressure', 'Temperature', 'S/C', 'ddG', '_Alkene_Type'])

# # Handle missing values by imputing with the mean
# imputer = SimpleImputer(strategy='mean')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Fit the scaler on the data
# scaler.fit(df_imputed)

# # Transform the data using the fitted scaler
# df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

# # Apply VarianceThreshold to remove low-variance features
# selector = VarianceThreshold(threshold=0)
# df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# # Apply K-means clustering with different numbers of clusters
# n_clusters = 8  # You can try values from 5 to 12
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(df_selected)

# # Apply PCA to the selected features for visualization
# pca = PCA(n_components=2)
# pca_results = pca.fit_transform(df_selected)

# # Create a DataFrame with the PCA results and cluster labels
# pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
# pca_df['Cluster'] = cluster_labels

# # Plot the PCA results with clusters
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=pca_df, s=100, legend='full')

# plt.title(f'PCA of Alkenes with K-means Clustering (k={n_clusters})')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()

# # If you want to visualize with different cluster numbers, loop through a range
# for n_clusters in range(5, 13):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(df_selected)
#     pca_df['Cluster'] = cluster_labels
    
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=pca_df, s=100, legend='full')

#     plt.title(f'PCA of Alkenes with K-means Clustering (k={n_clusters})')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.legend(title='Cluster')
#     plt.grid(True)
#     plt.show()




# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/reordered_final_merged_cleaned_qqm_molecule_charges.xlsx'
# df = pd.read_excel(file_path)

# # Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
# labels = df['Rxn ID']
# alkene_types = df['_Alkene_Type']

# # Drop irrelevant features and non-numeric columns, but keep '_Alkene_Type'
# df = df.drop(columns=['Rxn ID', 'Pressure/atm', 'Temperature/C', 'S/C', 'ddG'])

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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = '/Users/zhangjiayi/Desktop/Chem Research/molli/Practice/machine_learning/reordered_final_merged_cleaned_qqm_molecule_charges.xlsx'
df = pd.read_excel(file_path)

# Preserve the 'Reaction ID' and '_Alkene_Type' for labeling
labels = df['Rxn ID']
alkene_types = df['_Alkene_Type']

# Drop irrelevant features and non-numeric columns, but keep '_Alkene_Type'
df = df.drop(columns=['Rxn ID', 'Pressure/atm', 'Temperature/C', 'S/C', 'ddG'])

# Select only numeric columns
df_numeric = df.select_dtypes(include=[float, int])

# Drop rows with any missing values
df_numeric = df_numeric.dropna()

# Ensure 'alkene_types' aligns with the numeric dataframe
alkene_types = alkene_types[df_numeric.index]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the data
scaler.fit(df_numeric)

# Transform the data using the fitted scaler
df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)

# Apply VarianceThreshold to remove low-variance features
selector = VarianceThreshold(threshold=0)
df_selected = pd.DataFrame(selector.fit_transform(df_scaled), columns=df_scaled.columns[selector.get_support()])

# Apply PCA to Reactant Features with 3 components
reactant_features = [col for col in df_selected.columns if 'Q' in col or 'C' in col]
pca_reactants = PCA(n_components=3)
reactant_pca = pca_reactants.fit_transform(df_selected[reactant_features])

# Convert the PCA-transformed features back to a DataFrame
reactant_pca_df = pd.DataFrame(reactant_pca, columns=[f'Reactant_PCA{i+1}' for i in range(reactant_pca.shape[1])])

# Unique alkene types
unique_alkene_types = alkene_types.unique()

# Create a consistent color palette for both the points and the legend
palette = sns.color_palette("viridis", n_colors=len(unique_alkene_types))
color_mapping = {alkene_type: color for alkene_type, color in zip(unique_alkene_types, palette)}

# Visualization of PCA for Reactant Features colored by _Alkene_Type in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the points with consistent colors
for alkene_type in unique_alkene_types:
    indices = alkene_types == alkene_type
    # Ensure indices are used correctly for selection
    ax.scatter(reactant_pca_df.loc[indices, 'Reactant_PCA1'], 
               reactant_pca_df.loc[indices, 'Reactant_PCA2'], 
               reactant_pca_df.loc[indices, 'Reactant_PCA3'], 
               color=color_mapping[alkene_type], label=alkene_type, s=50)

ax.set_title('PCA of Reactant Features Colored by _Alkene_Type')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

# Adding legend with correct color mapping
ax.legend(title="_Alkene_Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
