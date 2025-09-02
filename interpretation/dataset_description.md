### Datasets

**Labeled-Motifs**
Labeled-Motifs is a synthetic dataset consisting of 2,000 graphs. Each graph is composed of a base motif (tree) and an additional motif, which can be either a tree, grid, or hexagon structure. To better emulate the real-world molecular datasets, where nodes in the functional group often exhibit distinct labels compared to skeletal nodes, we introduce node labels corresponding to their respective motifs. The grid motifs and hexagon motifs are designated as the ground truth explanations, with molecules containing these motifs used as test data. 

 **Mutagenicity**
Mutagenicity is a collection of 4,337 graphs for molecular property prediction. The graphs are classified into two different classes according to the mutagenicity of a molecule which is correlated with nitro groups (e.g. -NO<sub>2</sub>) and amino groups (e.g. -NH$_2$). We use molecules with -NO$_2$ or -NH$_2$ as test data, as they are the only samples with ground truth explanations (-NO$_2$ or -NH$_2$).
 
 **Solubility**
Solubility contains 1,144 molecules with different levels of aqueous solubility. Molecules with a log solubility value below -4 are labeled as 0, while those with values above -2 are labeled as 1. The hydroxyl groups, i.e. -OH, are treated as the ground truth explanation for solubility. We use molecules with -OH as test data.

 **Benzene**
Benzene consists of 12,000 molecules, which can be categorized into two distinct classes based on the presence or absence of the benzene structure within the molecule (excluding hydrogen atoms). We use molecules with benzene structure as test data.
