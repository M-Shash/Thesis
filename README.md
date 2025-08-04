# FedCAPE: Federated Concept-based Automated Post-hoc Explanations

FedCAPE is an open-source framework for federated concept discovery and automated post-hoc explanation of deep vision models. It enables scalable, privacy-preserving visual concept analysis and explanation across distributed clients, inspired by the automatic concept-based explanations paradigm. This implementation leverages modern vision backbone models (SAM2, CLIP, DINOv2, Inception-v3), advanced segmentation/filtering, federated K-Means clustering, and TCAV concept attribution, all orchestrated via the [Flower](https://flower.dev/) federated learning framework. This repository includes scripts and utilities for running large-scale distributed experiments on high-performance computing clusters such as Tartu HPC.

## Key Features

- **End-to-end local-to-global concept discovery** using federated clustering.
- **Privacy-preserving:** No raw images or labels are shared between clients and server; only cluster-level statistics and signatures are communicated.
- **Modern vision backbones:** Utilizes SAM2 for segmentation, CLIP for semantic filtering, DINOv2 for SSL embedding, and Inception-v3 for concept attribution.
- **Quantitative explanation:** Concept Activation Vectors (CAVs) and TCAV scores for model interpretability.
- **Rich visualization:** Exemplar overlays, UMAP, and heatmaps for cluster/concept analysis.
- **HPC/large-scale support:** Parallel, batched computation and resource management for multi-node clusters.

## Repository Structure

├── segment_dataset/
│ └── segments/ # Per-client segment and metadata storage
│     └── precomputed_client_0.pkl   -| 
│     └── precomputed_client_1.pkl    | the pickled sentimental segments along with their features after the Dinov2 
│     └── precomputed_client_2.pkl   -|
│     └── segments_client_0       -|
│     └── segments_client_1        | holds the segmentation for each and every image and their meta data -like segment locations- along with their segments is pickled in form of  precomputed_client_{clientId}.pkl
│     └── segments_client_2       -|
├── root/  # holds class directory for random images to help instantiate the centriods
│ └── basketball
│ └── mountain bike
│ └── moving van
├── test/ # holds the test images per class that will be used for TCAV scoring 
│ └── basketball
│ └── mountain bike
│ └── moving van
├── cluster_visualizations/ # Output for cluster/TCAV visualizations
├── data/
│ └──  ILSVRC2012
|    └── segmentation # The directory that actually holds the class images from which the code consumes and accumelate in ~/segment_dataset/segments/segments_clients{clientId}  
|       └──basketball
|       └──mountain bike
|       └──moving van
├── requirments.txt (for adjusting the virtual environment )
├── README.md # Project documentation (this file)
├── thesis-fed # Main implementation file (Python)

basicall python 3.12 is used.
ps the dirctory names should match the class name.
the Data is from Imagene Large Scale Visual Recognition Challenge 2012 (ILSVRC2012).  
and the logit class nuron number https://gist.github.com/xkumiyu/dd200f3f51986888c9151df4f2a9ef30#file-ilsvrc2012_classlist-txt-L191 
