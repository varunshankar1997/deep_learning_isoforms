# deep_learning_isoforms

Project of group from DTU about designing a  VAE to predict protein isoforms from gene expression data.  (By Christina Christiansen, Stefanos Rodopoulos and Varun Shankar)

The dataset used for the above projects are available from the following repositories. 
Owing to their sizes, the links from where one can obtain the same have been provided 
GTEX Dataset -> https://www.science.org/doi/10.1126/science.aaz1776
ARCHS4 dataset -> https://maayanlab.cloud/archs4/index.html


Since a large part of the computational work was performed on the HPC, the provided code should ideally be run on an HPC, preferably on an LSF-10 cluster. 
The datasets can also be accessed by :- cd /dtu-compute/datasets/iso_02456/

code for training and generating the VAE :- 
 bsub < jobscript_to_run_VAE.sh 
 
 
 All other program code is used to create the PCA along with performing the visualizations, which can be run locally on a machine
 
 




