
def eval_distortion(full_df:pd.DataFrame, fs_pipe:Pipeline, max_clusters=None, name='test', plot_distortion=False):

    inertias = list()
    distortions = list()

    if max_clusters:
        K = range(1, max_clusters+1)
    else:
        K = range(1, full_df.shape[0]+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(fs_pipe)
        #This returns the distortion (average euclidean squared distance from the centroid of the respective clusters)
        distortions.append(sum(np.min(cdist(fs_pipe, kmeanModel.cluster_centers_, 'euclidean'),axis=1)**2) / fs_pipe.shape[0])

        #This returns the inertia (sum of squared distances of samples to their closest cluster center)
        inertias.append(kmeanModel.inertia_)
    
    if plot_distortion:
        fig = plt.figure()
        #Based on Distortion
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.title(f'Distortion Plot vs. Clusters {name}')
        fig.savefig(f'{name}_distortion.png')

        # fig = plt.figure()
        # #Based on Inertia
        # plt.plot(K, inertias, 'bx-')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Inertia')
        # plt.title(f'Inertia Plot vs. Clusters {name}')
        # fig.savefig(f'{name}_inertia.png')
    return (list(K),distortions)

'''
this function measures the average of the squared distances from the points to their respective cluster centers. 
lower distortion, better
this function generates the distortion values for different numbers of clusters (K).
It plots these distortion values as a function of the number of clusters.
The kneed package is designed to automatically detect the "elbow" or "knee" point in a curve

the function measures the average squared distance from the points to their respective cluster center(k-means)
Then, it computes the distorstion values for a specified range of clusters.Then make a plot to see how the distortion changes as teh number of clusters increases. 
'''
