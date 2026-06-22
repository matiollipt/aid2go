# Aid2GO GO Graph Module
# Author: Cleverson Matiolli, Ph.D.

import math
import os
import pickle
import re
import logging
import requests
import torch
import numpy as np
import pandas as pd
import hdbscan
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
import plotly_express as px
import networkx as nx
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from obonet import read_obo
from torch_geometric.data import Data, HeteroData
from transformers import BertTokenizer, BertModel
from itertools import product
from collections import defaultdict

# Config logging parameters
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/aid2go.log"), logging.StreamHandler()],
)

# ==========================
# Gene Ontology Data Loader
# ==========================


def get_godag(output_dir="./data/go", timeout=30):
    """
    Downloads and loads the Gene Ontology (GO) graph in OBO format from O.

    Parameters:
    - save_path (str): Directory to save the OBO file and resulting data.
    - timeout (int): Maximum time to wait for download.

    Returns:
    - go_dag (nx.DiGraph): GO DAG.
    - go_df (pd.DataFrame): Node attributes in a DataFrame.
    """

    logging.info("Fetching GO DAG...")

    # Make dirs and set GO source
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / "go-basic.obo"
    
    url = "https://purl.obolibrary.org/obo/go/go-basic.obo"  # permanent URL (PURL)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    except requests.RequestException as e:
        logging.error(f"Failed to download GO DAG: {e}")
        raise RuntimeError(f"Failed to download GO DAG: {e}")

    # Parse and extract node's attributes
    go_dag = read_obo(filepath, ignore_obsolete=True)
    nodes_data = [
        {
            "term": term,
            "name": attrs.get("name", ""),
            "namespace": attrs.get("namespace", "unknown"),
            "definition": (
                attrs.get("def", "").split('"', 2)[1] if "def" in attrs else ""
            ),  # The split is to prevent GO term identifiers from leaking in training data
        }
        for term, attrs in go_dag.nodes(data=True)
    ]

    go_df = pd.DataFrame(nodes_data)
    go_df["in_degree"] = go_df["term"].map(dict(go_dag.in_degree))
    go_df["out_degree"] = go_df["term"].map(dict(go_dag.out_degree))
    go_df["degree"] = go_df["in_degree"] + go_df["out_degree"]
    go_df.to_csv(os.path.join(output_dir, "go-basic.csv"), index=False)

    logging.info(f"GO DAG and nodes' attributes (dataframe) saved in {output_dir}")

    return go_dag, go_df


# =======================
# GO Relationships Parser
# =======================


def parse_go_relationships(go_dag, save_path="./data/go"):
    """
    Parses relationships from the GO DAG.

    Parameters:
    - go_dag (nx.DiGraph): GO DAG.
    - save_path (str): Directory to save relationships.

    Returns:
    - go_edges_df (pd.DataFrame): DataFrame of relationships.
    """
    logging.info("Parsing GO relationships...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    edges = [
        {"source_term": u, "relationship": d, "target_term": v}
        for u, v, d in go_dag.edges
    ]

    go_edges_df = pd.DataFrame(edges)
    go_edges_df.to_csv(save_path / "go_edges.tsv", sep="\t", index=False)

    logging.info(f"GO relationships saved to {save_path / 'go_edges.tsv'}")
    return go_edges_df


# =====================
# Embedding GO Terms
# =====================


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, text_column):
        self.texts = dataframe[text_column].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def embed_go_terms(
    df,
    column_id,
    column_text,
    batch_size=32,
    model_name="bert-base-uncased",
    save_path="./data/go",
):
    """
    Embeds GO terms using a pre-trained BERT model.

    Parameters:
    - df (pd.DataFrame): DataFrame containing GO terms.
    - column_id (str): Column with GO IDs.
    - column_text (str): Column with GO term definitions.
    - batch_size (int): Batch size for embedding.
    - model_name (str): Pre-trained model name.
    - save_path (str): Directory to save embeddings.

    Returns:
    - embeddings_dict (dict): Dictionary of embeddings.
    """
    logging.info("Starting embedding of GO terms...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    dataset = TextDataset(df, column_text)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    embeddings_dict = {}
    for texts in tqdm(dataloader, desc="Embedding texts"):
        inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        for text, emb in zip(texts, outputs.cpu().numpy()):
            term = df.loc[df[column_text] == text, column_id].iloc[0]
            embeddings_dict[term] = emb

    with open(save_path / "go_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)

    logging.info(f"GO embeddings saved to {save_path / 'go_embeddings.pkl'}")
    return embeddings_dict


# =========================
# GO Graph Data Conversion
# =========================


def create_go_graph_data(go_edges_df, go_embed_dict, save_path, multi_edge=False):
    """
    Converts GO relationships and embeddings into graph data.

    Parameters:
    - go_edges_df (pd.DataFrame): DataFrame of GO relationships.
    - go_embed_dict (dict): GO embeddings.
    - save_path (str): Path to save graph data.
    - multi_edge (bool): Whether to create heterogeneous graph data.

    Returns:
    - graph_data (HeteroData or Data): PyTorch Geometric data object.
    """
    logging.info("Creating GO graph data...")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    node_features = torch.stack([torch.tensor(v) for v in go_embed_dict.values()])
    node_mapping = {k: i for i, k in enumerate(go_embed_dict.keys())}

    edge_index = (
        torch.tensor(
            [
                [node_mapping[row["source_term"]], node_mapping[row["target_term"]]]
                for _, row in go_edges_df.iterrows()
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )

    if multi_edge:
        data = HeteroData()
        data["go"].x = node_features
        for rel in go_edges_df["relationship"].unique():
            rel_edges = go_edges_df[go_edges_df["relationship"] == rel]
            rel_edge_index = (
                torch.tensor(
                    [
                        [
                            node_mapping[row["source_term"]],
                            node_mapping[row["target_term"]],
                        ]
                        for _, row in rel_edges.iterrows()
                    ],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
            data["go", rel, "go"].edge_index = rel_edge_index
        torch.save(data, save_path / "go_data_multiedge.pt")
        logging.info(
            f"Heterogeneous GO graph data saved to {save_path / 'go_data_multiedge.pt'}"
        )
    else:
        data = Data(x=node_features, edge_index=edge_index)
        torch.save(data, save_path / "go_data.pt")
        logging.info(f"Homogeneous GO graph data saved to {save_path / 'go_data.pt'}")

    return data

################################################################
# Visualization functions
################################################################

def plot_word_count_distribution(go_df, save_path):
    """
    Plots the word count distribution in the given GO text data using a kernel density estimation plot.

    Parameters:
    go_df (pandas.DataFrame): The DataFrame containing the GO text data. It should have a column named 'def_word_count' representing the word count of each definition text.
    save_path (pathlib.Path): The path where the generated plot will be saved.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        go_df["def_word_count"],
        color="blue",
        label="Definition Texts",
        fill=True,
        alpha=0.5,
    )
    plt.title("Word Count Distribution in GO Text Data")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "word_counts.png", dpi=300)
    plt.show()

def plot_degree_ia_distribution(go_df, save_path):
    """
    Plots the distribution of degree and Information Accretion (IA) values from a given DataFrame.

    Parameters:
    go_df (pd.DataFrame): The DataFrame containing the degree and IA values.
    save_path (pathlib.Path): The path where the generated plot will be saved.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting the distribution of degree
    sns.histplot(go_df["degree"].unique(), bins=50, kde=True, ax=ax[0], color="blue")
    ax[0].set_title("Distribution of Degree")
    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Frequency")

    # Plotting the distribution of IA values
    sns.histplot(go_df["ia"], bins=50, kde=True, ax=ax[1], color="green")
    ax[1].set_title("Distribution of IA Values")
    ax[1].set_xlabel("IA Value")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path / "degrees_ia_distribution.png", dpi=300)
    plt.show()


def plot_degree_vs_ia(go_df, save_path):
    """
    Plots Kernel Density Estimation (KDE) plots of the Information Accretion (IA) values against the in-degree, out-degree, and total degree of the GO terms.

    Parameters:
    go_df (pandas.DataFrame): A DataFrame containing the GO terms and their corresponding in_degree, out_degree, and IA values.
    save_path (pathlib.Path): The path where the generated plots will be saved.

    Returns:
    None
    """
    # Calculate total degree
    go_df['degree'] = go_df['in_degree'] + go_df['out_degree']

    # List of degree types to plot
    degree_types = ['in_degree', 'out_degree', 'degree']

    for degree_type in degree_types:
        plt.figure(figsize=(14, 10))
        sns.kdeplot(
            data=go_df,
            x=degree_type,
            y="ia",
            cmap="coolwarm",
            fill=True,
            cbar=True,
            cbar_kws={"label": "Density"},
        )
        plt.xlim(0, 25)
        plt.title(f"KDE Plot of IA vs {degree_type.capitalize().replace('_', ' ')}")
        plt.xlabel(degree_type.capitalize().replace('_', ' '))
        plt.ylabel("IA Value")
        plt.tight_layout()
        plt.savefig(save_path / f"{degree_type}_vs_ia_heatmap.png", dpi=300)
        plt.show()
        plt.close()

    print(f"Degree vs IA plots saved in: {save_path}")


# def analyze_clusters(
#     go_sample, go_terms, num_of_neighbors, save_path: Path, silhouette_threshold: float
# ):
#     """
#     Analyzes clusters for a given sample using HDBSCAN with various min_cluster_size values,
#     saves the plots to the specified path, and returns the data points with silhouette score
#     above and below the specified threshold.

#     Parameters:
#     go_sample (np.ndarray): The sample data for clustering.
#     go_terms (list): List of GO terms corresponding to the data points in go_sample.
#     num_of_neighbors (list): List of min_cluster_size values to try.
#     save_path (Path): Path object where plots will be saved.
#     silhouette_threshold (float): The threshold for silhouette scores to filter data points.

#     Returns:
#     dict: Contains results for each min_cluster_size, with data points above and below the silhouette score threshold.
#     """
#     results = {}
#     noisy_terms = {}

#     for n in num_of_neighbors:
#         high_silhouette_data_points = []
#         low_silhouette_data_points = []

#         # Create a subplot with 1 row and 2 columns
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         fig.set_size_inches(18, 7)

#         # Initialize the HDBSCAN clusterer
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=n)
#         cluster_labels = clusterer.fit_predict(go_sample)

#         # Filter out noise points (cluster_label == -1)
#         filtered_indices = np.where(cluster_labels != -1)[0]
#         filtered_labels = cluster_labels[filtered_indices]
#         filtered_X = go_sample[filtered_indices]
#         filtered_go_terms = np.array(go_terms)[filtered_indices]
        

#         # Identify noisy points from the ORIGINAL dataset (labels == -1)
#         noisy_indices = np.where(cluster_labels == -1)[0]
#         noisy_go_terms = np.array(go_terms)[noisy_indices]

#         print(f"# neighbors = {n}, # noisy points = {len(noisy_indices)}")

#         # Store the noisy points (from original data) for later reference
#         noisy_terms[n] = {
#             "noisy_points": noisy_go_terms,
#             "filtered_points": filtered_go_terms,
#         }

#         # Save just the noisy GO terms array (or you can save the entire dict if desired)
#         np.save(save_path / f"noisy_points_{n}.npy", noisy_go_terms)

#         # If all points are noisy, skip silhouette calculations
#         if len(filtered_labels) == 0:
#             print(f"# neighbors = {n}, all points are considered noise.")
#             plt.close()
#             continue

#         # The silhouette_score gives the average value for all the samples.
#         silhouette_avg = silhouette_score(filtered_X, filtered_labels)
#         print(
#             f"# neighbors = {n}, # clusters = {len(np.unique(filtered_labels))} | Mean Silhouette Score = {silhouette_avg:.2f}"
#         )

#         # Compute the silhouette scores for each sample
#         sample_silhouette_values = silhouette_samples(filtered_X, filtered_labels)

#         y_lower = 10
#         for i in range(len(np.unique(filtered_labels))):
#             # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
#             ith_cluster_silhouette_values = sample_silhouette_values[
#                 filtered_labels == i
#             ]

#             ith_cluster_silhouette_values.sort()

#             size_cluster_i = ith_cluster_silhouette_values.shape[0]
#             y_upper = y_lower + size_cluster_i

#             color = mcolors.nipy_spectral(float(i) / len(np.unique(filtered_labels)))
#             ax1.fill_betweenx(
#                 np.arange(y_lower, y_upper),
#                 0,
#                 ith_cluster_silhouette_values,
#                 facecolor=color,
#                 edgecolor=color,
#                 alpha=0.7,
#             )

#             # Label the silhouette plots with their cluster numbers at the middle
#             ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#             # Compute the new y_lower for next plot
#             y_lower = y_upper + 10  # 10 for the 0 samples

#         ax1.set_title("Silhouette Plot (Clusters)")
#         ax1.set_xlabel("Silhouette Coefficient Values")
#         ax1.set_ylabel("Cluster label")

#         # The vertical line for average silhouette score of all the values
#         ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#         ax1.set_yticks([])  # Clear the yaxis labels / ticks
#         ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#         # 2nd Plot showing the actual clusters formed
#         colors = mcolors.nipy_spectral(
#             filtered_labels.astype(float) / len(np.unique(filtered_labels))
#         )
#         ax2.scatter(
#             filtered_X[:, 0],
#             filtered_X[:, 1],
#             marker=".",
#             s=30,
#             lw=0,
#             alpha=0.7,
#             c=colors,
#             edgecolor="k",
#         )

#         # Labeling the clusters
#         centers = clusterer.exemplars_
#         if centers is not None:
#             centers = np.array([np.mean(center, axis=0) for center in centers])
#             # Draw white circles at cluster centers
#             ax2.scatter(
#                 centers[:, 0],
#                 centers[:, 1],
#                 marker="o",
#                 c="white",
#                 alpha=1,
#                 s=200,
#                 edgecolor="k",
#             )

#             for i, c in enumerate(centers):
#                 ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

#         ax2.set_title("Clustered Data Visualization")
#         ax2.set_xlabel("Feature space (1st feature)")
#         ax2.set_ylabel("Feature space (2nd feature)")

#         plt.suptitle(
#             "Silhouette analysis for HDBSCAN clustering on sample data with min_cluster_size = %d"
#             % n,
#             fontsize=14,
#             fontweight="bold",
#         )

#         # Save the plot
#         save_file_path = save_path / f"cluster_analysis_min_cluster_size_{n}.png"
#         plt.savefig(save_file_path, dpi=300)
#         plt.show()
#         plt.close()

#         # Get the data points with silhouette score above and below the specified threshold
#         for point, label, silhouette, go_term in zip(
#             filtered_X, filtered_labels, sample_silhouette_values, filtered_go_terms
#         ):
#             data_point_info = {
#                 "data_point": point,
#                 "cluster_label": label,
#                 "silhouette_score": silhouette,
#                 "go_term": go_term,
#             }
#             if silhouette > silhouette_threshold:
#                 high_silhouette_data_points.append(data_point_info)
#             else:
#                 low_silhouette_data_points.append(data_point_info)

#         results[n] = {
#             "above_threshold": high_silhouette_data_points,
#             "below_threshold": low_silhouette_data_points,
#         }

#     return results, noisy_terms

def analyze_clusters(
    go_sample, go_terms, num_of_neighbors, save_path: Path, silhouette_threshold: float
):
    """
    Analyzes clusters for a given sample using HDBSCAN with various min_cluster_size values,
    saves the plots to the specified path, and returns the data points with silhouette score
    above and below the specified threshold.

    Parameters:
    go_sample (np.ndarray): The sample data for clustering.
    go_terms (list): List of GO terms corresponding to the data points in go_sample.
    num_of_neighbors (list): List of min_cluster_size values to try.
    save_path (Path): Path object where plots will be saved.
    silhouette_threshold (float): The threshold for silhouette scores to filter data points.

    Returns:
    dict: Contains results for each min_cluster_size, with data points above and below the silhouette score threshold.
    """
    results = {}
    noisy_terms = {}

    for n in num_of_neighbors:
        high_silhouette_data_points = []
        low_silhouette_data_points = []

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(min_cluster_size=n)
        cluster_labels = clusterer.fit_predict(go_sample)

        # Filter out noise points (cluster_label == -1)
        filtered_indices = np.where(cluster_labels != -1)[0]
        filtered_labels = cluster_labels[filtered_indices]
        filtered_X = go_sample[filtered_indices]
        filtered_go_terms = np.array(go_terms)[filtered_indices]
        

        # Identify noisy points from the ORIGINAL dataset (labels == -1)
        noisy_indices = np.where(cluster_labels == -1)[0]
        noisy_go_terms = np.array(go_terms)[noisy_indices]

        print(f"# neighbors = {n}, # noisy points = {len(noisy_indices)}")

        # Store the noisy points (from original data) for later reference
        noisy_terms[n] = {
            "noisy_points": noisy_go_terms,
            "filtered_points": filtered_go_terms,
        }

        # Save just the noisy GO terms array (or you can save the entire dict if desired)
        np.save(save_path / f"noisy_points_{n}.npy", noisy_go_terms)

        # If all points are noisy, skip silhouette calculations
        if len(filtered_labels) == 0:
            print(f"# neighbors = {n}, all points are considered noise.")
            plt.close()
            continue

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(filtered_X, filtered_labels)
        print(
            f"# neighbors = {n}, # clusters = {len(np.unique(filtered_labels))} | Mean Silhouette Score = {silhouette_avg:.2f}"
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(filtered_X, filtered_labels)

        # Get colormap for silhouette plotting
        color_map = plt.get_cmap("nipy_spectral")
        
        y_lower = 10
        for i in range(len(np.unique(filtered_labels))):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                filtered_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i


            color = color_map(float(i) / len(np.unique(filtered_labels)))  # Correct way
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette Plot (Clusters)")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        unique_labels = np.unique(filtered_labels)
        color_map = plt.get_cmap("nipy_spectral")
        colors = [color_map(float(label) / len(unique_labels)) for label in filtered_labels]  # Generate colors per cluster

        ax2.scatter(
            filtered_X[:, 0],
            filtered_X[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.exemplars_
        if centers is not None:
            centers = np.array([np.mean(center, axis=0) for center in centers])
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("Clustered Data Visualization")
        ax2.set_xlabel("Feature space (1st feature)")
        ax2.set_ylabel("Feature space (2nd feature)")

        plt.suptitle(
            "Silhouette analysis for HDBSCAN clustering on sample data with min_cluster_size = %d"
            % n,
            fontsize=14,
            fontweight="bold",
        )

        # Save the plot
        save_file_path = save_path / f"cluster_analysis_min_cluster_size_{n}.png"
        plt.savefig(save_file_path, dpi=300)
        plt.show()
        plt.close()

        # Get the data points with silhouette score above and below the specified threshold
        for point, label, silhouette, go_term in zip(
            filtered_X, filtered_labels, sample_silhouette_values, filtered_go_terms
        ):
            data_point_info = {
                "data_point": point,
                "cluster_label": label,
                "silhouette_score": silhouette,
                "go_term": go_term,
            }
            if silhouette > silhouette_threshold:
                high_silhouette_data_points.append(data_point_info)
            else:
                low_silhouette_data_points.append(data_point_info)

        results[n] = {
            "above_threshold": high_silhouette_data_points,
            "below_threshold": low_silhouette_data_points,
        }

    return results, noisy_terms



def perform_kmeans_elbow(data, max_clusters=15):
    """
    Perform KMeans clustering with elbow method to determine the best number of clusters.

    Args:
    data (array-like): The input data for clustering.
    max_clusters (int): The maximum number of clusters to consider.

    Returns:
    KElbowVisualizer: The fitted KElbowVisualizer object.
    """
    model = KElbowVisualizer(KMeans(n_init="auto"), k=max_clusters)
    model.fit(data)
    return model


def perform_hdbscan(data, min_cluster_size=10, min_samples=30):
    """
    Perform HDBSCAN clustering.

    Args:
    data (array-like): The input data for clustering.
    min_cluster_size (int): The minimum size of clusters.
    min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    tuple: Cluster labels and filtered data (noise points removed).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    cluster_labels = clusterer.fit_predict(data)

    mask = cluster_labels != -1
    filtered_labels = cluster_labels[mask]
    filtered_data = data[mask]

    return filtered_labels, filtered_data


def plot_clusters(tsne_results, labels, title, ax):
    """
    Plot clusters using t-SNE results.

    Args:
    tsne_results (array-like): The t-SNE results.
    labels (array-like): The cluster labels or categories.
    title (str): The title of the plot.
    ax (matplotlib.axes.Axes): The axes to plot on.
    """
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", n_colors=len(unique_labels))
    for label, color in zip(unique_labels, palette):
        mask = labels == label
        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=[color],
            label=f"Cluster {label}",
            s=50,
            alpha=0.7,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE feature 1")
    ax.set_ylabel("t-SNE feature 2")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def plot_silhouette(data, labels, ax):
    """
    Plot silhouette analysis for clustering results.

    Args:
    data (array-like): The input data used for clustering.
    labels (array-like): The cluster labels.
    ax (matplotlib.axes.Axes): The axes to plot on.

    Returns:
    float: The average silhouette score.
    """
    silhouette_avg = silhouette_score(data, labels)
    sample_silhouette_values = silhouette_samples(data, labels)

    y_lower = 10
    for i in range(len(np.unique(labels))):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = mcolors.nipy_spectral(float(i) / len(np.unique(labels)))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette plot for the clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return silhouette_avg


def perform_umap(
    data,
    n_components=3,
    n_neighbors_list=[3, 5, 15, 30],
    min_dist_list=[0.0, 0.2, 0.99],
    metrics=["cosine", "euclidean", "manhattan"],
    n_epochs=[1000],
    learning_rate=1,
):
    """
    Perform UMAP dimensionality reduction with grid search.

    Args:
    data (array-like): The input data for dimensionality reduction.
    n_components (int): The number of components in the embedded space.
    n_neighbors_list (list): List of n_neighbors values to try.
    min_dist_list (list): List of min_dist values to try.
    metrics (list): List of distance metrics to try.
    n_epochs (list): List of n_epochs values to try.
    learning_rate (float): The learning rate for the optimization.

    Returns:
    dict: A dictionary containing UMAP results for different parameter combinations.
    """
    umap_dict = {}
    parameter_combinations = product(n_neighbors_list, min_dist_list, metrics, n_epochs)

    for n_neighbors, min_dist, metric, epoch in tqdm(
        parameter_combinations, desc="UMAP Parameter Search"
    ):
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            n_epochs=epoch,
            learning_rate=learning_rate,
            min_dist=min_dist,
            verbose=False,
            random_state=42,
        )
        results = reducer.fit_transform(data)
        umap_dict.update({f"{n_neighbors}_{min_dist}_{metric}_{epoch}": results})

    return umap_dict


def perform_spectral_embedding(
    data, n_components=3, n_neighbors_list=[3, 5, 10, 20, 30, 50, 100]
):
    """
    Perform Spectral Embedding dimensionality reduction with different n_neighbors values.

    Args:
    data (array-like): The input data for dimensionality reduction.
    n_components (int): The number of components in the embedded space.
    n_neighbors_list (list): List of n_neighbors values to try.

    Returns:
    dict: A dictionary containing Spectral Embedding results for different n_neighbors values.
    """
    se_dict = {}

    for n_neighbor in tqdm(n_neighbors_list, desc="Searching n_neighbors..."):
        se = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbor,
            affinity="rbf",
            random_state=42,
            n_jobs=-1,
        )
        results = se.fit_transform(data)
        se_dict.update({n_neighbor: results})

    return se_dict


def perform_tsne(data, n_components=2, random_state=42):
    """
    Perform t-SNE dimensionality reduction.

    Args:
    data (array-like): The input data for dimensionality reduction.
    n_components (int): The number of components in the embedded space.
    random_state (int): Random state for reproducibility.

    Returns:
    array: The t-SNE results.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(data)


def perform_tsne_grid_search(
    data,
    perplexities=[5, 30, 50, 100],
    n_iter_list=[250, 500, 1000],
    learning_rates=[10, 200, 1000],
    n_components=3,
):
    """
    Perform t-SNE dimensionality reduction with grid search over hyperparameters.

    Args:
    data (array-like): The input data for dimensionality reduction.
    perplexities (list): List of perplexity values to try.
    n_iter_list (list): List of number of iterations to try.
    learning_rates (list): List of learning rates to try.
    n_components (int): The number of components in the embedded space.

    Returns:
    dict: A dictionary containing t-SNE results for different parameter combinations.
    """
    tsne_dict = {}
    parameter_combinations = product(perplexities, n_iter_list, learning_rates)

    for perplexity, n_iter, learning_rate in tqdm(
        list(parameter_combinations), desc="t-SNE Grid Search"
    ):
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=42,
        )
        results = tsne.fit_transform(data)
        tsne_dict[f"perp_{perplexity}_iter_{n_iter}_lr_{learning_rate}"] = results

    return tsne_dict


def create_tsne_dataframe(tsne_dict, original_df):
    """
    Create a DataFrame from t-SNE results for easy plotting and analysis.

    Args:
    tsne_dict (dict): Dictionary containing t-SNE results from grid search.
    original_df (pd.DataFrame): Original DataFrame with metadata.

    Returns:
    pd.DataFrame: DataFrame containing t-SNE results and original metadata.
    """
    df_tsne = pd.DataFrame()

    for key, value in tsne_dict.items():
        df_temp = pd.DataFrame(
            value, columns=[f"tsne_{i+1}_{key}" for i in range(value.shape[1])]
        )
        df_tsne = pd.concat([df_tsne, df_temp], axis=1)

    # Combine with original metadata
    df_combined = pd.concat([original_df.reset_index(drop=True), df_tsne], axis=1)

    return df_combined


def plot_3d_tsne(
    df,
    perplexity,
    n_iter,
    learning_rate,
    color_column,
    hover_data=None,
    width=800,
    height=600,
    size=2,
    opacity=0.7,
):
    """
    Create an interactive 3D scatter plot of t-SNE results using Plotly.

    Args:
    df (pd.DataFrame): The dataframe containing the t-SNE results and metadata.
    perplexity (float): The perplexity value to plot.
    n_iter (int): The number of iterations to plot.
    learning_rate (float): The learning rate to plot.
    color_column (str): The column name for color-coding the points.
    hover_data (list): List of column names to include in hover information.
    width (int): Width of the plot.
    height (int): Height of the plot.
    size (int): Size of the markers.
    opacity (float): Opacity of the markers.

    Returns:
    plotly.graph_objs._figure.Figure: The interactive 3D scatter plot.
    """
    key = f"perp_{perplexity}_iter_{n_iter}_lr_{learning_rate}"
    x = f"tsne_1_{key}"
    y = f"tsne_2_{key}"
    z = f"tsne_3_{key}"

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=df[color_column],
        labels={"color": color_column},
        hover_data=hover_data,
    )
    fig.update_traces(marker=dict(size=size), opacity=opacity)
    fig.update_layout(
        width=width,
        height=height,
        title=f"t-SNE 3D Plot (Perplexity: {perplexity}, Iterations: {n_iter}, Learning Rate: {learning_rate})",
    )
    return fig

def get_node_depths(G: nx.DiGraph()) -> dict:
    """
    Compute the depth of each node in the graph based on topological generations.
    
    Args:
    G (nx.DiGraph): The directed graph representing the GO hierarchy.
    
    Returns:
    dict: A dictionary mapping each node to its depth in the graph.
    """
    node_depths = {}
    for depth, generation in enumerate(reversed(list(nx.topological_generations(G)))):
        for node in generation:
            node_depths[node] = depth
    return node_depths

def analyze_go_graph_topology(G: nx.DiGraph(), plot: bool = True):
    """
    Analyze the topological structure of the GO graph and optionally plot the distribution.
    
    Args:
    G (nx.DiGraph): The directed graph representing the GO hierarchy.
    plot (bool): Whether to plot the distribution of generations by namespace.
    
    Returns:
    tuple: (node_depths, generation_counts)
        node_depths (dict): A dictionary mapping each node to its depth in the graph.
        generation_counts (dict): A dictionary containing the count of nodes in each generation by namespace.
    """
    node_depths = get_node_depths(G)
    
    # Count nodes in each generation by namespace
    generation_counts = defaultdict(lambda: defaultdict(int))
    for node, depth in node_depths.items():
        namespace = G.nodes()[node]["namespace"]
        generation_counts[depth][namespace] += 1
    
    # Print generations (limited to 10 items per generation)
    for depth, nodes in enumerate(reversed(list(nx.topological_generations(G)))):
        nodes_info = [f"{node} ({G.nodes()[node]['namespace']})" for node in nodes[:10]]
        # print(f"Generation {depth} -> {nodes_info}{' ...' if len(nodes) > 10 else ''}")
    
    if plot:
        plot_generation_distribution(generation_counts)
    
    return node_depths, dict(generation_counts)

def plot_generation_distribution(generation_counts: dict):
    """
    Plot the distribution of generations by namespace.
    
    Args:
    generation_counts (dict): A dictionary containing the count of nodes in each generation by namespace.
    """
    depths = sorted(generation_counts.keys())
    namespaces = set(namespace for counts in generation_counts.values() for namespace in counts)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.25
    index = range(len(depths))
    
    for i, namespace in enumerate(namespaces):
        counts = [generation_counts[depth].get(namespace, 0) for depth in depths]
        ax.bar([x + i*bar_width for x in index], counts, bar_width, label=namespace)
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Distribution of GO Terms by Depth and Namespace')
    ax.set_xticks([x + bar_width for x in index])
    ax.set_xticklabels(depths)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

