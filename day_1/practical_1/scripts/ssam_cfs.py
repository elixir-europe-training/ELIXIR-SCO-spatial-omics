import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssam
import json
import pickle
import os
import shutil
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import to_rgba, to_hex
from matplotlib_venn import venn2
import time
from matplotlib.colors import ListedColormap
import re

####################custom functions####################
def cc_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"{directory_path} found and removed.")
    os.makedirs(directory_path)
    print(f"{directory_path} created.")

def extract_gene_panel_info(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    rows = []

    for target in data['payload']['targets']:
        target_type = target.get('type', {}).get('data', {})
        target_descriptor = target.get('type', {}).get('descriptor', '')

        if target_descriptor == 'gene':
            rows.append({
                'Ensembl ID': target_type.get('id', ''),
                'Name': target_type.get('name', ''),
                'Annotation': target_descriptor
            })

    df = pd.DataFrame(rows)
    return df

def save_pickle(fn, o):
    with open(fn, "wb") as f:
        return pickle.dump(o, f, protocol=4)
    
def load_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

def plot_diagnostic_plot(self, centroid_index, cluster_name=None, cluster_color=None, cmap=None, rotate=0, z=None, use_embedding="tsne", known_signatures=[], correlation_methods=[]):
    """
    Plot the diagnostic plot. This method requires `plot_tsne` or `plot_umap` was run at least once before.

    :param centroid_index: Index of the centroid for the diagnostic plot.
    :type centroid_index: int
    :param cluster_name: The name of the cluster.
    :type cluster_name: str
    :param cluster_color: The color of the cluster. Overrides `cmap` parameter.
    :type cluster_color: str or list(float)
    :param cmap: The colormap for the clusters. The cluster color is determined using the `centroid_index` th color of the given colormap.
    :type cmap: str or matplotlib.colors.Colormap
    :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
    :type rotate: int
    :param z: Z index to slice 3D vector norm and cell-type map plots.
        If not given, the slice at the middle will be used.
    :type z: int
    :param use_embedding: The type of the embedding for the last panel. Possible values are "tsne" or "umap".
    :type use_embedding: str
    :param known_signatures: The list of known signatures, which will be displayed in the 3rd panel. Each signature can be 3-tuple or 4-tuple,
        containing 1) the name of signature, 2) gene labels of the signature, 3) gene expression values of the signature, 4) optionally the color of the signature.
    :type known_signatures: list(tuple)
    :param correlation_methods: The correlation method used to determine max correlation of the centroid to the `known_signatures`. Each method should be 2-tuple,
        containing 1) the name of the correaltion, 2) the correaltion function (compatiable with the correlation methods available in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_)
    :type correlation_methods: list(tuple)
    """
    if z is None:
        z = int(self.vf_norm.shape[2] / 2)
    p, e = self.centroids[centroid_index], self.centroids_stdev[centroid_index]
    if cluster_name is None:
        cluster_name = "Cluster #%d"%centroid_index
    
    if cluster_color is None:
        if cmap is None:
            cmap = plt.get_cmap("jet")
        cluster_color = cmap(centroid_index / (len(self.centroids) - 1))

    if len(correlation_methods) == 0:
        correlation_methods = [("r", corr), ]
    total_signatures = len(correlation_methods) * len(known_signatures) + 1
            
    ax = plt.subplot(1, 4, 1)
    mask = self.filtered_cluster_labels == centroid_index
    plt.scatter(self.local_maxs[0][mask], self.local_maxs[1][mask], c=[cluster_color])
    self.plot_l1norm(rotate=rotate, cmap="Greys", z=z)

    ax = plt.subplot(1, 4, 2)
    ctmap = np.zeros([self.filtered_celltype_maps.shape[0], self.filtered_celltype_maps.shape[1], 4])
    ctmap[self.filtered_celltype_maps[..., z] == centroid_index] = to_rgba(cluster_color)
    ctmap[np.logical_and(self.filtered_celltype_maps[..., z] != centroid_index, self.filtered_celltype_maps[..., 0] > -1)] = [0.9, 0.9, 0.9, 1]
    if rotate == 1 or rotate == 3:
        ctmap = ctmap.swapaxes(0, 1)
    ax.imshow(ctmap)
    if rotate == 1:
        ax.invert_xaxis()
    elif rotate == 2:
        ax.invert_xaxis()
        ax.invert_yaxis()
    elif rotate == 3:
        ax.invert_yaxis()

    ax = plt.subplot(total_signatures, 4, 3)
    ax.bar(self.genes, p, yerr=e)
    ax.set_title(cluster_name)
    plt.xlim([-1, len(self.genes)])
    plt.xticks(rotation=90, fontsize=6.5)

    subplot_idx = 0
    for signature in known_signatures:
        sig_title, sig_labels, sig_values = signature[:3]
        sig_colors_defined = False
        if len(signature) == 4:
            sig_colors = signature[3]
            sig_colors_defined = True
        for corr_label, corr_func in correlation_methods:
            corr_results = [corr_func(p, sig_value) for sig_value in sig_values]
            corr_results = [e[0] if hasattr(e, "__getitem__") else e for e in corr_results]
            max_corr_idx = np.argmax(corr_results)
            ax = plt.subplot(total_signatures, 4, 7+subplot_idx*4)
            lbl = sig_labels[max_corr_idx]
            if sig_colors_defined:
                col = sig_colors[max_corr_idx]
            else:
                col = cluster_color
            ax.bar(self.genes, sig_values[max_corr_idx], color=col)
            ax.set_title("%s in %s (max %s, %.3f)"%(lbl, sig_title, corr_label, corr_results[max_corr_idx]))
            plt.xlim([-1, len(self.genes)])
            plt.xticks(rotation=90, fontsize=6.5)
            subplot_idx += 1

    if use_embedding == 'tsne':
        embedding = self.tsne
        fig_title = "t-SNE, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
    elif use_embedding == 'umap':
        embedding = self.umap
        fig_title = "UMAP, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
    good_vectors = self.filtered_cluster_labels[self.filtered_cluster_labels != -1]
    ax = plt.subplot(1, 4, 4)
    ax.scatter(embedding[:, 0][good_vectors != centroid_index], embedding[:, 1][good_vectors != centroid_index], c=[[0.8, 0.8, 0.8, 1],], s=80)
    ax.scatter(embedding[:, 0][good_vectors == centroid_index], embedding[:, 1][good_vectors == centroid_index], c=[cluster_color], s=80)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(fig_title)

def calculate_correlation_matrix(ds_centroids, scrna_centroids, scrna_uniq_labels, save_filename=None):
    num_ds_centroids = len(ds_centroids)
    num_scrna_centroids = len(scrna_centroids)
    correlation_matrix = np.zeros((num_ds_centroids, num_scrna_centroids))
    
    for i in range(num_ds_centroids):
        for j in range(num_scrna_centroids):
            p = ds_centroids[i]
            scrna_centroid = scrna_centroids[j]
            corr_result_pear, pval_pear = pearsonr(p, scrna_centroid)
            corr_result_spear, pval_spear = spearmanr(p, scrna_centroid)
            corr_result = max(corr_result_pear, corr_result_spear)
            # Store all correlation results for the current pair of centroids
            correlation_matrix[i, j] = corr_result
    
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(len(scrna_centroids), len(scrna_centroids)))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')

    # Set y-axis ticks and labels
    ax.set_yticks(np.arange(num_ds_centroids))
    ax.set_yticklabels([f'Cluster #{i}' for i in range(num_ds_centroids)])

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(num_scrna_centroids))
    ax.set_xticklabels(scrna_uniq_labels, rotation=90)

    plt.xlabel("scRNA-seq Centroids")
    plt.ylabel("Xenium Centroids")
    plt.title("Correlation Heatmap")

    cbar = fig.colorbar(cax, shrink=0.6) 

    if save_filename:
        plt.savefig(save_filename, dpi=300)

    plt.show()
    return correlation_matrix

def plot_celltypes_map_and_localmax(self, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, scatter_color=None, scatter_cmap=None, scatter_size=1, color_labels=None):
    """
    Plot the merged cell-type map and scatter plot the local maxima.
    This function combines the functionalities of plot_celltypes_map and plot_localmax.

    Parameters:

    """

    # Code from plot_celltypes_map method
    if z is None:
        z = int(self.shape[2] / 2)
    num_ctmaps = np.max(self.filtered_celltype_maps) + 1
    
    if len(centroid_indices) == 0:
        centroid_indices = list(range(num_ctmaps))
        
    if colors is None:
        cmap_internal = plt.get_cmap(cmap)
        colors = cmap_internal([float(i) / (num_ctmaps - 1) for i in range(num_ctmaps)])

        
    all_colors = [background if not j in centroid_indices else colors[i] for i, j in enumerate(range(num_ctmaps))]
    cmap_internal = ListedColormap(all_colors)

    celltype_maps_internal = np.array(self.filtered_celltype_maps[..., z], copy=True)
    empty_mask = celltype_maps_internal == -1
    celltype_maps_internal[empty_mask] = 0
    sctmap = cmap_internal(celltype_maps_internal)
    sctmap[empty_mask] = (0, 0, 0, 0)

    if set_alpha:
        alpha = np.array(self.max_correlations[..., z], copy=True)
        alpha[alpha < 0] = 0
        alpha = min_r + alpha / (np.max(alpha) / (1.0 - min_r))
        sctmap[..., 3] = alpha

    if rotate == 1 or rotate == 3:
        sctmap = sctmap.swapaxes(0, 1)

    plt.gca().set_facecolor(background)
    plt.imshow(sctmap)
    
    if rotate == 1:
        plt.gca().invert_xaxis()
    elif rotate == 2:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    elif rotate == 3:
        plt.gca().invert_yaxis()

    # Code from plot_localmax method
    if rotate < 0 or rotate > 3:
        raise ValueError("rotate can only be 0, 1, 2, 3")
    if rotate == 0 or rotate == 2:
        dim0, dim1 = 1, 0
    elif rotate == 1 or rotate == 3:
        dim0, dim1 = 0, 1
    plt.scatter(self.local_maxs[dim0], self.local_maxs[dim1], s=scatter_size, c=scatter_color, cmap=scatter_cmap)
    plt.xlim([0, self.vf_norm.shape[dim0]])
    plt.ylim([self.vf_norm.shape[dim1], 0])
    if rotate == 1:
        plt.gca().invert_xaxis()
    elif rotate == 2:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    elif rotate == 3:
        plt.gca().invert_yaxis()

    plt.gca().invert_yaxis()
    
    # Create a color guide legend
    if colors is not None and color_labels is not None:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in zip(colors, color_labels)]
        plt.legend(handles=legend_elements, loc='upper right')

    #plt.show()

def load_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

def cc_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def process_cells_link(link):
    data = []
    match = re.search(r'&target=([\d._]+)&', link)
    if match:
        coordinates = match.group(1).split('_')
        x_coord, y_coord = coordinates[0], coordinates[1]
        data.append({'X_coordinate': x_coord, 'Y_coordinate': y_coord})
    else:
        print('No X and Y coordinates found in link.')
    cell_info = pd.DataFrame(data)
    return cell_info


def create_toy_dataset(df, toy_percent, toy_parameter, gene_panel=None):
    print(f"Creating toy dataset with {toy_percent}% of {toy_parameter}s from the original dataset...")
    org_im_width = df.x_location.max() - df.x_location.min()
    org_im_height = df.y_location.max() - df.y_location.min()
    if toy_parameter == "coordinate":
        toy_df = df.sample(frac=toy_percent/100, random_state=1)
    elif toy_parameter == "gene":
        gene_panel = gene_panel.sample(frac=toy_percent/100, random_state=1)
        toy_df = df[df.feature_name.isin(gene_panel['Name'])]
    elif toy_parameter == "pixel":
        toy_im_width = org_im_width * (toy_percent/100)
        toy_im_height = org_im_height * (toy_percent/100)
        # Filter original dataset to get the toy dataset
        toy_df = df[
            (df.x_location >= ((org_im_width/2) - (toy_im_width/2))) &
            (df.x_location <= ((org_im_width/2) + (toy_im_width/2))) &
            (df.y_location >= ((org_im_height/2) - (toy_im_height/2))) &
            (df.y_location <= ((org_im_height/2) + (toy_im_height/2)))
        ].copy()
    elif toy_parameter == "euclidean":
        x0, y0 = org_im_width/2, org_im_height/2
        x, y = df.x_location, df.y_location
        deuclidean_distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        max_distance = deuclidean_distance.max()
        toy_df = df[deuclidean_distance <= (max_distance * (toy_percent/100))].copy()
        
    # Make move the toy dataset from the center of the original image to the top left corner
    toy_df['x_location'] = toy_df['x_location'] - toy_df['x_location'].min()
    toy_df['y_location'] = toy_df['y_location'] - toy_df['y_location'].min()

    print(f"Original image width: {org_im_width}")
    print(f"Original image height: {org_im_height}")
    print(f"toy image width: {np.ceil(toy_df.x_location.max() - toy_df.x_location.min())}")
    print(f"toy image height: {np.ceil(toy_df.y_location.max() - toy_df.y_location.min())}")
    print(f"Number of unique genes in the toy dataset: {len(set(toy_df['feature_name'].unique()))}")
    print(f"Number of coordinates in the toy dataset: {len(toy_df)}")

    return toy_df




#################### test ####################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_celltypes_map_and_localmax_with_bg(self, image_path, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None, scatter_color=None, scatter_cmap=None, scatter_size=1, color_labels=None, scale_factor=1.0):
    # Load the image
    img = mpimg.imread(image_path)
    #img = np.rot90(img, k=rotate)
    img = np.flipud(img)
    new_extent = [0, img.shape[1] * scale_factor, 0, img.shape[0] * scale_factor]

    if z is None:
        z = int(self.shape[2] / 2)
    num_ctmaps = np.max(self.filtered_celltype_maps) + 1
    
    if len(centroid_indices) == 0:
        centroid_indices = list(range(num_ctmaps))
        
    if colors is None:
        cmap_internal = plt.get_cmap(cmap)
        colors = cmap_internal([float(i) / (num_ctmaps - 1) for i in range(num_ctmaps)])

        
    all_colors = [background if not j in centroid_indices else colors[i] for i, j in enumerate(range(num_ctmaps))]
    cmap_internal = ListedColormap(all_colors)

    celltype_maps_internal = np.array(self.filtered_celltype_maps[..., z], copy=True)
    empty_mask = celltype_maps_internal == -1
    celltype_maps_internal[empty_mask] = 0
    sctmap = cmap_internal(celltype_maps_internal)
    sctmap[empty_mask] = (0, 0, 0, 0)

    if set_alpha:
        alpha = np.array(self.max_correlations[..., z], copy=True)
        alpha[alpha < 0] = 0
        alpha = min_r + alpha / (np.max(alpha) / (1.0 - min_r))
        sctmap[..., 3] = alpha

    if rotate == 1 or rotate == 3:
        sctmap = sctmap.swapaxes(0, 1)

    plt.gca().set_facecolor(background)

    # Plot the image as the background
    plt.imshow(img, extent=new_extent, aspect='auto')

    plt.imshow(sctmap)
    
    if rotate == 1:
        plt.gca().invert_xaxis()
    elif rotate == 2:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    elif rotate == 3:
        plt.gca().invert_yaxis()

    # Code from plot_localmax method
    if rotate < 0 or rotate > 3:
        raise ValueError("rotate can only be 0, 1, 2, 3")
    if rotate == 0 or rotate == 2:
        dim0, dim1 = 1, 0
    elif rotate == 1 or rotate == 3:
        dim0, dim1 = 0, 1
    plt.scatter(self.local_maxs[dim0], self.local_maxs[dim1], s=scatter_size, c=scatter_color, cmap=scatter_cmap)
    plt.xlim([0, self.vf_norm.shape[dim0]])
    plt.ylim([self.vf_norm.shape[dim1], 0])
    if rotate == 1:
        plt.gca().invert_xaxis()
    elif rotate == 2:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    elif rotate == 3:
        plt.gca().invert_yaxis()

    # Create a color guide legend
    if colors is not None and color_labels is not None:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in zip(colors, color_labels)]
        plt.legend(handles=legend_elements, loc='upper right')
