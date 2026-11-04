The dataset (PubChemElements_all.csv) has 118 elements and eight numeric features: AtomicMass, Electronegativity, AtomicRadius, IonizationEnergy, ElectronAffinity, MeltingPoint, BoilingPoint and Density. Many heavy elements have blanks, YearDiscovered contains the text ``Ancient'', and StandardState contains values such as ``Expected to be a Solid''. Handle each of these explicitly.

 

Q1. PCA is very popularly used for feature selection. Discuss a feature selection method presented in the attached paper. Implement the method to rank the eight features.

 

Q2. It is surmised that the 8 features are connected with the location of the elements in the periodic table. Use the k-means algorithm to cluster the data and find if the elements in the various clusters are linked with proximity in the periodic table.
