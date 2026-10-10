import os

import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT = "output"
FASTA_PATH = "data/globin_family.fasta"
LABELS_PATH = "data/known_subfamilies.csv"




####### LOAD SEQUENCES #######

def load_sequences(fasta_path):
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        ungapped = str(record.seq).replace("-", "").upper()
        records[record.id] = ungapped
    return records




#### FEATURES ######

def sequence_features(sequence):
    analysis = ProteinAnalysis(sequence)
    features = dict(analysis.amino_acids_percent)  
    features["length"] = len(sequence)
    features["molecular_weight"] = analysis.molecular_weight()
    features["aromaticity"] = analysis.aromaticity()
    features["isoelectric_point"] = analysis.isoelectric_point()
    features["gravy"] = analysis.gravy()
    return features


def build_feature_table(sequences):
    rows = {seq_id: sequence_features(seq) for seq_id, seq in sequences.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "sequence_id"
    return df





##### K-MEANS SILHOUETTE SCORE #####

def choose_k_by_silhouette(pca, k_range):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=1, n_init=10).fit_predict(pca)
        scores[k] = silhouette_score(pca, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores





###### MAIN #######

def main():
    os.makedirs(OUTPUT, exist_ok=True)

    sequences = load_sequences(FASTA_PATH)

    feature_df = build_feature_table(sequences)
    feature_df.to_csv(f"{OUTPUT}/features_table.csv")

    labels_df = pd.read_csv(LABELS_PATH, index_col="sequence_id")
    labels_df = labels_df.loc[feature_df.index] 
    true_family = labels_df["known_subfamily"]

    # Standardize since features are on different scales 
    # (Molecular weight of globins are in the thousands whilst
    # amino acid percentages are <100%)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df.values)


    pca_full = PCA()
    pca = pca_full.fit_transform(scaled)
    pca_2d = PCA(n_components=2)
    pca2 = pca_2d.fit_transform(scaled)





    # Choose k using silhouette method
    k_range = range(2, 8)
    best_k, silhouette_scores = choose_k_by_silhouette(pca, k_range)
    n_known_families = true_family.nunique()
    print(f"\nBest k by silhouette score: {best_k} (there are {n_known_families} known subfamilies).")

    plt.figure(figsize=(6, 4))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o")
    plt.axvline(n_known_families, color="grey", linestyle="--", label=f"{n_known_families} known subfamilies")
    plt.axvline(best_k, color="grey", linestyle=":", label="Silhouette optimal")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Choosing k for K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/silhouette_scores.png", dpi=150)
    plt.close()





    # Table comparing K-means clustering with silhouette optimal k clusters vs. true number of subfamilies
    kmeans_labels = {}
    rows = []
    for name, k in [("silhouette_optimal", best_k), ("true_subfamilies", n_known_families)]:
        kmeans_labels[name] = KMeans(n_clusters=k, random_state=1, n_init=10).fit_predict(pca)
        ari = adjusted_rand_score(true_family, kmeans_labels[name])
        nmi = normalized_mutual_info_score(true_family, kmeans_labels[name])
        rows.append([name, k, ari, nmi])
    print("\nSimilarity between K-Means clustering vs. actual known subfamilies:")
    print(pd.DataFrame(rows, columns=["clustering", "k", "ARI", "NMI"]).to_string(index=False))



    # PCA plot coloured based on silhouette optimal K-Means clusters
    plt.figure(figsize=(7, 6))
    for cluster_id in sorted(set(kmeans_labels["silhouette_optimal"])):
        mask = kmeans_labels["silhouette_optimal"] == cluster_id
        plt.scatter(pca2[mask, 0], pca2[mask, 1], label=f"Cluster {cluster_id}", s=90)
    for i, seq_id in enumerate(feature_df.index):
        plt.annotate(seq_id, (pca2[i, 0], pca2[i, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"K-Means clusters (k={best_k}) in PCA space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/pca_by_silhouette_optimal_kmeans_cluster.png", dpi=150)
    plt.close()

    # PCA plot coloured based on K-Means clusters of K=5 known subfamilies 
    plt.figure(figsize=(7, 6))
    for cluster_id in sorted(set(kmeans_labels["true_subfamilies"])):
        mask = kmeans_labels["true_subfamilies"] == cluster_id
        plt.scatter(pca2[mask, 0], pca2[mask, 1], label=f"Cluster {cluster_id}", s=90)
    for i, seq_id in enumerate(feature_df.index):
        plt.annotate(seq_id, (pca2[i, 0], pca2[i, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(f"K-Means clusters (k={n_known_families}) in PCA space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/pca_by_true_subfamilies_kmeans_cluster.png", dpi=150)
    plt.close()



    # PCA plot coloured based on true known subfamilies 
    plt.figure(figsize=(7, 6))
    for family in sorted(true_family.unique()):
        mask = (true_family == family).values
        plt.scatter(pca2[mask, 0], pca2[mask, 1], label=family, s=90)
    for i, seq_id in enumerate(feature_df.index):
        plt.annotate(seq_id, (pca2[i, 0], pca2[i, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("True globin subfamilies in PCA space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/pca_by_true_family.png", dpi=150)
    plt.close()






    # Hierarchical clustering dendrogram   
    linkage_matrix = linkage(pca, method="average")
    leaf_labels = [f"{seq_id} ({true_family[seq_id]})" for seq_id in feature_df.index]
    plt.figure(figsize=(9, 6))
    dendrogram(linkage_matrix, labels=leaf_labels, leaf_font_size=9, orientation="right")
    plt.title("Hierarchical clustering of human globin family")
    plt.xlabel("Distance (PCA space)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/dendrogram.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
