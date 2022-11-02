# PUMP (Predicting Underspecification Monitoring Pipeline)
### J. Tang, M. Kressman, H. Lakshmankumar, B. Aduaka, A. Jakusovszky, P. Anderson, J. Davidson
Department of Computer Science, California Polytechnic University, San Luis Obispo,California, USA 

Department of Biological Sciences, California Polytechnic University, San Luis Obispo,California, USA

## The Problem

Underspecification is an issue experienced by deep learning networks that describes when a model's training is unable to be accurately applied to datasets beyond the data it was trained on. If a model outputs a high quantity when describing topics like classification accuracy on a specifc dataset, then one would expect it to perform well on another, however this is not always the case and can be problematic, especially when applied to biomedicine.

## The Tool

PUMP is an open-source package that measures underspecification. The package was developed using datasets regarding breast canver subtyping based on gene expression.

## Our Dataset

The use-case utilized was a transcriptomic METABRIC dataset with 19,084 gene expressions and 2,133 patient samples. 

This dataset has more features than samples. This is a characteristic of many datasets at risk of underspecification.

Find this data set here:

`brca_metabric_clinical_data.tsv`

## Code
1. `underspecification.py`: contains class UnderspecificationAnalysis that handles data preprocessing and analysis.
2. `pipeline.ipynb`: Notebook for user to iteratively setup and track progress of underspecifaction analysis on desired data that meets format needs.

In addition to code files, this repository also includes results images, libraries like pydeep2 which are used for analysis, and some source data. 

The source code was developed using a Jupyter Notebook environment. Find instructions to install Jupyter here: [Installing Jupyter using pip](https://jupyter.org/install)
