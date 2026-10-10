# K-Means clustering of the human globin family

Can K-Means clustering of sequence features separate the human globin subfamilies 
without using sequence alignment?

## Data

15 human globin protein sequences from NCBI RefSeq. Accessions and known subfamilies are in
`data/known_subfamilies.csv`.

## Method

1. Read the FASTA file with Biopython and remove alignment gaps.
2. Calculate features for each sequence using Biopython's `ProteinAnalysis` function:
   amino acid composition, length, molecular weight, aromaticity, isoelectric point
   and GRAVY hydropathy.
3. Standardize the features using `StandardScaler` and plot PCA of the features.
4. Run K-Means with k = 2-7 and pick the k with the highest silhouette score.
   Also run K-Means with k = 5 which is the number of known subfamilies.
5. Compare both clusterings to the known subfamilies using the Adjusted Rand Index (ARI)
   and Normalized Mutual Information (NMI).
6. Plot a hierarchical clustering dendrogram of the same PCA features as
   another way to see how the sequences relate.

## Results

With k = 3, the globins split into hemoglobins, myoglobins and the cellular globins 
(cytoglobin and neuroglobin). Whilst k = 5, shows the same clustering as the true subfamilies.

The silhouette scores are low (~0.2-0.3) and close together across different k
number of clusters, so the data don't point strongly to any one number of clusters.

## Limitations

- Amino acid composition as features for K-Means clustering fails to account for 
  residue order. Whilst protein families are usually grouped by sequence similarity
  from sequence alignments, which does account for residue order.