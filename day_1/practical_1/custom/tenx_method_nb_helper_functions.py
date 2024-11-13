import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, argrelextrema
from scipy.sparse import issparse, csr_matrix
from diptest import diptest
import statsmodels.api as sm
from shapely.geometry import Point, Polygon, MultiPoint
from scipy.spatial import ConvexHull
import re

####################### Helper functions####################
def create_adata(sample_path, nucleus_genes_only=False):
    """
    Create an AnnData object from the 10x data files.

    Parameters:
    - sample_path: Path to the sample data folder.
    - nucleus_genes_only: Boolean indicating whether to only consider transcripts within the nucleolus.

    Returns:
    - adata: AnnData object created using the provided sample data.
    """
    # Construct the necessary file paths
    cell_feature_matrix_path = os.path.join(sample_path, "cell_feature_matrix.h5")
    cells_csv_path = os.path.join(sample_path, "cells.csv")
    cells_df = pd.read_csv(cells_csv_path)

    if nucleus_genes_only:
        # Construct the necessary file paths
        transcripts_csv_path = os.path.join(sample_path, "transcripts.csv")
        features_gz_path = os.path.join(sample_path, "cell_feature_matrix","features.tsv.gz")

        # Load additional file for nucleus consideration
        features_df = pd.read_csv(features_gz_path, sep='\t', compression='gzip', header=None)
        transcripts_df = pd.read_csv(transcripts_csv_path)
        
        # Filter for real genes in the assay
        genes = features_df[features_df[2] == "Gene Expression"][1].values

        # Remove transcripts that are not genes and not in the gene list
        transcripts_df_genes = transcripts_df[transcripts_df["feature_name"].isin(genes)]

        # Remove transcripts with qz<20
        transcripts_df_qv = transcripts_df_genes[transcripts_df_genes["qv"]>=20]

        # Remove unassigned transcripts
        transcripts_df_assigned = transcripts_df_qv[transcripts_df_qv["cell_id"] != "UNASSIGNED"]

        # Only keep transcripts that overlaps_nucleus is 1
        new_transcripts_df = transcripts_df_assigned[transcripts_df_assigned["overlaps_nucleus"] == 1]

        # Group by cell_id and count number of transcripts
        transcripts_df_assigned_overlaps_nucleus_grouped = new_transcripts_df.groupby("cell_id").size().reset_index(name="transcripts_count_nucleus")

        # Merge cells_df with transcripts_df_assigned_overlaps_nucleus_grouped
        cells_df_merged = pd.merge(cells_df, transcripts_df_assigned_overlaps_nucleus_grouped, on="cell_id", how="left")

        # Remove transcript_counts column and rename transcripts_count_nucleus as transcript_counts
        new_cells_df = cells_df_merged.drop(columns=["transcript_counts"])
        new_cells_df = new_cells_df.rename(columns={"transcripts_count_nucleus": "transcript_counts"})

        # Create a count matrix based on the new_transcripts_df
        count_matrix = new_transcripts_df.pivot_table(index='cell_id', columns='feature_name', aggfunc='size', fill_value=0)

        # Align cells_df with the count matrix by setting index and reindexing
        new_cells_df = new_cells_df.set_index('cell_id').reindex(count_matrix.index)

        # Create AnnData object
        adata = sc.AnnData(X=count_matrix.values, obs=new_cells_df, var=pd.DataFrame(index=count_matrix.columns))

        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].to_numpy()
    else:
        adata = sc.read_10x_h5(cell_feature_matrix_path)
        cells_df.set_index(adata.obs_names, inplace=True)
        adata.obs = cells_df.copy()
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].to_numpy()
            
    return adata

# Function to convert shapely polygons to a format storable in AnnData
def polygon_to_coords(polygon):
    if polygon.is_empty:
        return None
    else:
        return list(polygon.exterior.coords)
    
def calculate_cell_boundaries(adata, transcripts_df):
    """
    Calculates cell boundaries from transcript data and updates the AnnData object.

    Parameters:
    - adata: AnnData object to be updated.
    - transcripts_df: pandas DataFrame containing transcript coordinates and cell IDs.

    Returns:
    - Updated AnnData object with cell boundaries stored in adata.uns['cell_boundaries'].
    """

    # Group by cell_id and calculate the boundary
    cell_boundaries = {}
    grouped = transcripts_df.groupby('cell_id')
    
    for cell_id, group in grouped:
        points = group[['x_location', 'y_location']].values
        if len(points) < 3:
            continue  # We need at least 3 points to calculate a boundary
        polygon = MultiPoint(points).convex_hull  # Use convex_hull to get the outer boundary
        cell_boundaries[cell_id] = polygon
        
    # Convert polygons to coordinate lists
    cell_boundaries_coords = {k: polygon_to_coords(v) for k, v in cell_boundaries.items()}
    
    # Add cell boundaries to AnnData
    adata.uns['cell_boundaries'] = cell_boundaries_coords
    
    return adata

def calculate_nucleus_boundaries(adata, nucleus_df):
    """
    Calculates nucleus boundaries from nucleus coordinates data and updates the AnnData object.

    Parameters:
    - adata: AnnData object to be updated.
    - nucleus_df: pandas DataFrame containing nucleus coordinates.

    Returns:
    - Updated AnnData object with nucleus boundaries stored in adata.uns['nucleus_boundaries'].
    """
    # Ensure that vertex_x and vertex_y are treated as numeric
    nucleus_df['vertex_x'] = pd.to_numeric(nucleus_df['vertex_x'])
    nucleus_df['vertex_y'] = pd.to_numeric(nucleus_df['vertex_y'])
    
    # Group by cell_id and create a Polygon for each nucleus
    nucleus_polygons = nucleus_df.groupby('cell_id').apply(
        lambda group: Polygon(zip(group['vertex_x'], group['vertex_y']))
    ).to_dict()

    # Convert polygons to coordinate lists
    nucleus_boundaries_coords = {k: polygon_to_coords(v) for k, v in nucleus_polygons.items()}

    # Update adata with nucleus boundaries
    adata.uns['nucleus_boundaries'] = nucleus_boundaries_coords
    
    return adata


# Function to calculate the bandwidth for kernel density estimation based on Silverman's rule of thumb
def calculate_bandwidth_silverman(data):
    return 1.06 * data.std() * len(data) ** (-1 / 5.)

def kde_cv(x, bandwidths):
    scores = []
    for bandwidth in bandwidths:
        # Compute cross validation score for each bandwidth
        score = compute_loo_score(x, bandwidth)
        scores.append(score)
    best_bandwidth = bandwidths[np.argmin(scores)]
    return best_bandwidth, scores

def compute_loo_score(x, bandwidth):
    scores = []
    for i in range(len(x)):
        # Leave one out: use all data points except the ith point
        training_data = np.delete(x, i)
        validation_point = x[i]
        
        # Create KDE with given bandwidth
        kde = gaussian_kde(training_data, bw_method=bandwidth)
        
        # Evaluate the KDE on the left out point
        estimated_density = kde.evaluate([validation_point])[0]
        
        # Actual density if using all data for estimation (this is a bit tricky, usually we don't do this step in LOO)
        kde_all = gaussian_kde(x, bw_method=bandwidth)
        actual_density = kde_all.evaluate([validation_point])[0]
        
        # Compute a score (squared error in this case)
        scores.append((estimated_density - actual_density) ** 2)

    # Return the mean of all squared errors
    return np.mean(scores)

# Function to perform Hartigans' Dip Test for unimodality
def perform_dip_test(data):
    dip_statistic, p_value = diptest(data.values)
    return dip_statistic, p_value

def analyze_gene_expressions(adata, gene, bandwidth=0.3, plot=True, filter_zeros=False):
    x = adata[:, gene].X.toarray().flatten()
    if filter_zeros:
        x = x[x > 0]
    
    if len(x) > 1:
        density = gaussian_kde(x, bw_method=bandwidth)
        xgrid = np.linspace(min(x), max(x), 1000)

        dip, p_value = perform_dip_test(pd.Series(x))
        modality = 'Unimodal' if p_value > 0.05 else 'Multimodal'
        peaks, _ = find_peaks(density(xgrid), distance=20)
        minima = argrelextrema(density(xgrid), np.less)[0]

        highest_peak_xpos = None
        last_min_before_peak = None

        if peaks.size > 0:
            highest_peak = np.argmax(density(xgrid[peaks]))
            highest_peak_xpos = peaks[highest_peak]

        if highest_peak_xpos is not None and minima.size > 0:
            relevant_minima = minima[minima < highest_peak_xpos]
            if relevant_minima.size > 0:
                last_min_before_peak = relevant_minima[-1]

        if plot:
            plt.figure(figsize=(8, 4))
            plt.title(f'Expression Distribution for {gene}')
            plt.plot(xgrid, density(xgrid), label='Density')
            plt.hist(x, bins=50, alpha=0.3, density=True, label='Histogram')
            if highest_peak_xpos is not None:
                plt.axvline(x=xgrid[highest_peak_xpos], color='blue', linestyle=':', label='Highest Peak')
            if last_min_before_peak is not None:
                plt.axvline(x=xgrid[last_min_before_peak], color='red', linestyle='--', label='Minima before Peak')
            plt.xlabel('Expression Level')
            plt.ylabel('Density')
            plt.legend(title=f"Modality: {modality} (p-value={p_value:.2f})")
            plt.show()
            print(f"{gene} has a {modality} modality (p-value={p_value:.2f})")
            if modality == 'Multimodal':
                if last_min_before_peak is not None:
                    print(f"Background threshold for {gene} is {xgrid[last_min_before_peak]:0.3f}")

        if modality == 'Multimodal' and last_min_before_peak is not None:
            return xgrid[last_min_before_peak]

    return None

def plot_spatial_genes(adata, genes, cell_boundaries=False, nucleus_boundaries=False, cmap='light_dark_red'):
    """
    Plots the spatial distribution of transcripts and overlays cell and nucleus boundaries.

    Parameters:
    - adata: AnnData object containing spatial coordinates.
    - genes: List of genes to plot.
    - show_all_transcripts: Whether to show all transcripts or only those associated with cells.
    - cmap: Colormap to use for plotting (default is 'light_dark_blue').
    """
    plt.figure(figsize=(8, 6))
    plt.title('Spatial Distribution of Transcripts')

    if cell_boundaries:
        for cell_id, boundary in adata.uns['cell_boundaries'].items():
            if boundary is not None:
                coords = np.array(boundary)
                plt.plot(coords[:, 0], -coords[:, 1], color='blue', alpha=0.1)

    if nucleus_boundaries:
        for cell_id, boundary in adata.uns['nucleus_boundaries'].items():
            if boundary is not None:
                coords = np.array(boundary)
                plt.plot(coords[:, 0], -coords[:, 1], color='red', alpha=0.1)

    coordinates = adata.obsm['spatial']
        
    for gene in genes:
        if gene in adata.var_names:
            gene_expression = adata[:, gene].X
            if issparse(gene_expression):
                gene_expression = gene_expression.toarray().flatten()  # Ensure it's flattened

            gene_expression = gene_expression.flatten()  # Ensure it's flattened
            
            x = coordinates[:, 0]
            y = -coordinates[:, 1]

            # filter out zero expression values
            mask = gene_expression > 0
            x = x[mask]
            y = y[mask]
            gene_expression = gene_expression[mask]

            # Plotting the spatial distribution of transcripts
            plt.scatter(x, y, c=gene_expression, cmap=cmap, s=1, label=gene)  # Use custom colormap
            plt.colorbar(label='Gene Expression Level')

        else:
            print(f"{gene} gene is not found in the dataset.")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_spatial_transcripts(adata, transcripts_df, genes, cell_boundaries=False, nucleus_boundaries=False):
    """
    Plots the spatial distribution of transcripts and overlays cell and nucleus boundaries.
    
    Parameters:
    - adata: AnnData object containing spatial coordinates.
    - transcripts_df: DataFrame containing transcript information.
    - genes: List of genes to plot.
    - cell_boundaries: Whether to show cell boundaries.
    - nucleus_boundaries: Whether to show nucleus boundaries.
    """
    plt.figure(figsize=(8, 8))

    # Plot cell boundaries
    if cell_boundaries:
        # check if cell boundaries are present in adata.uns
        if 'cell_boundaries' not in adata.uns:
            print('Cell boundaries not found in adata.uns. Please run calculate_cell_boundaries() function first.')
        else:
            for cell_id, boundary in adata.uns['cell_boundaries'].items():
                if boundary is not None:
                    coords = np.array(boundary)
                    plt.plot(coords[:, 0], -coords[:, 1], color='blue', alpha=0.1)

    # Plot nucleus boundaries
    if nucleus_boundaries:
        # check if nucleus boundaries are present in adata.uns
        if 'nucleus_boundaries' not in adata.uns:
            print('Nucleus boundaries not found in adata.uns. Please run calculate_nucleus_boundaries() function first.')
        else:
            for cell_id, boundary in adata.uns['nucleus_boundaries'].items():
                if boundary is not None:
                    coords = np.array(boundary)
                    plt.plot(coords[:, 0], -coords[:, 1], color='gray', alpha=0.1)

    # Subsetting the DataFrame for the selected genes
    transcripts_df_4plot = transcripts_df[transcripts_df['feature_name'].isin(genes) & transcripts_df['cell_id'].isin(adata.obs_names)]

    # Get unique feature names
    feature_names = transcripts_df_4plot['feature_name'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(feature_names)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x'] * (len(feature_names) // 11 + 1)
    
    # Map feature names to colors and shapes
    feature_color_shape_map = {feature_name: (color, marker) for feature_name, color, marker in zip(feature_names, colors, markers[:len(feature_names)])}
    
    # Create the scatter plot for transcripts
    for feature_name in feature_names:
        subset = transcripts_df_4plot[transcripts_df_4plot['feature_name'] == feature_name]
        plt.scatter(
            subset['x_location'],
            -subset['y_location'],
            s=0.01,  # Adjust size as needed
            color=feature_color_shape_map[feature_name][0],  # color 
            marker=feature_color_shape_map[feature_name][1],  # shape
            label=feature_name
        )

    # Set plot labels and title
    plt.xlabel('x_location')
    plt.ylabel('y_location')
    plt.title('Spatial Distribution of Transcripts')

    # Create legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    plt.legend(*zip(*unique), markerscale=10)  # Increase markerscale if needed

    # Show the plot
    plt.show()

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