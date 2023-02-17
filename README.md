# Beta-sibling-study

The published code was used to obtain the MEG phenotype data describing sensorimotor spectral properties and beta event characteristics presented in the manuscript entitled ‘Human sensorimotor beta event characteristics and aperiodic signal are highly heritable’ by Pauls et al. available at https://biorxiv.org/cgi/content/short/2023.02.10.527950v1.

The analysis was carried out using MNE version 0.22.0, Python version 3.8.6, Numpy version 1.20.1, Scipy version 1.6, and Pandas version 1.2.2. The scripts are numbered according to the order in which they need to be used. There is a support function repository which contains the utility functions used in the analysis (for .fif data reading, amplitude envelope calculation, beta event detection, plotting etc.). Two meta data files which contain the subject metadata (age, gender, sibling structure information) and the manual phenotyping information used for analysis are also provided. 

The scripts expect the following folder structure in the main directory:
figures
metadata_tables
raw_data
python_scripts

