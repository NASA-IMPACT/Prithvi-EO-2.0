# Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications
This repository contains code and examples based on the [TerraTorch](github.com/IBM/terratorch) library for fine-tuning [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO), a more powerful version of the foundation model [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) developed by IBM and NASA, which offers significant improvements over its predecessor.
## Architecture Overview

Prithvi-EO-2.0 is based on the ViT architecture, pre-trained using a masked autoencoder (MAE) approach, with two major modifications as shown in the figure below. First, we introduce a random dropout mechanism that completely removes different bands before the patch embeddings, with the aim of improving the ability of the model to deal with missingness of data. Second, we make modifications to support inputs with temporal and multi-spectral characteristics. 
![model_architecture](https://github.com/user-attachments/assets/d9d52807-f7ca-48bc-b010-e5178f790155)

Our main modifications to the ViT architecture are the 3D positional embedding and the 3D patch embedding, which are required to deal with spatiotemporal data. We have also included metadata and process metadata about the actual geolocation (e.g. latitude and longitude) and date (i.e. year and day-of-year ranging 1-365). This is done by adding biases that are calculated via 2D sine-cosine positional encoding and added to the 3D positional embeddings and 3D patch embeddings via a learned weighted sum (i.e. the weight given is a parameter learned during pretraining). Since this metadata is often not available, we pretrained Prithvi-EO-2.0 allowing for this to be absent via a dropout.

## Pre-trained Models

| Model | Details | Weights |
| ------------- | ------------- | ------------- |
|Prithvi-EO-2.0-300M   | Pretrained 300M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_300M)  |
|Prithvi-EO-2.0-300M-TL   | Pretrained 300M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_300M_TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_300M_TL)  |
|Prithvi-EO-2.0-600M   | Pretrained 600M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_600M) |
|Prithvi-EO-2.0-600M-TL   | Pretrained 600M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_600M_TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-Granite-EO/Prithvi_EO_V2_600M_TL)   |


## Benchmarking

We used the most popular and rigorous benchmark framework available for Earth Observation foundation models: [GEO-Bench](https://github.com/ServiceNow/geo-bench). 

## Fine-tuning
