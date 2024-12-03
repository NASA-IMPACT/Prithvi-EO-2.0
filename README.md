# Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Prithvi-EO-2.0">
</p>

#### [Daniela Szwarcman](https://www.linkedin.com/in/daniela-szwarcman-60b55876/), [Sujit Roy](https://www.linkedin.com/in/sujit-roy01/), [Paolo Fraccaro](https://www.linkedin.com/in/paolo-fraccaro-3b85371b/?originalSubdomain=uk), [Þorsteinn Elí Gíslason](https://www.linkedin.com/in/%C3%BEorsteinn-el%C3%AD-g%C3%ADslason-a6ab951a9), [Benedikt Blumenstiel](https://www.linkedin.com/in/blumenstiel/), [Rinki Ghosal](https://www.linkedin.com/in/rinki-ghosal-5b2a41106/), [Pedro Henrique de Oliveira](https://www.linkedin.com/in/pedro-henrique-conrado-420377220/), [João Lucas de Sousa Almeida](https://www.linkedin.com/in/jo%C3%A3o-lucas-de-sousa-almeida-a08b9255/), [Rocco Sedona](https://www.linkedin.com/in/rocco-sedona-79812749/), [Yanghui Kang](https://www.linkedin.com/in/yanghui-kang-797aa33a/), [Srija Chakraborty](https://www.linkedin.com/in/chakrabortysrija/), [Sizhe Wang](https://scholar.google.com/citations?user=bucEAU0AAAAJ&hl=en), [Ankur Kumar](https://www.linkedin.com/in/ankurk017/), [Myscon Truong](https://www.linkedin.com/in/myscon-truong/), [Denys Godwin](https://www.linkedin.com/in/denys-godwin-43a49188/), [Hyunho Lee](https://scholar.google.com/citations?user=oOwJeyQAAAAJ), [Chia-Yu Hsu](https://www.linkedin.com/in/chiayu-hsu/), [Ata Akbari Asanjan](https://www.linkedin.com/in/ataakbariasanjan/), [Besart Mujeci](https://www.linkedin.com/in/besart/), [Trevor Keenan](https://www.linkedin.com/in/trevor-keenan/), [Paulo Arévolo](https://scholar.google.com/citations?user=AwYBme4AAAAJ&hl=en), [Wenwen Li](https://www.linkedin.com/in/wenwenli/), [Hamed Alemohammad](https://www.linkedin.com/in/hamedalemo/), [Pontus Olofsson](https://www.linkedin.com/in/pontus-olofsson-057701255/), [Christopher Hain](https://www.linkedin.com/in/christopher-hain-5b465917b/), [Robert Kennedy](https://scholar.google.com/citations?user=I-2_GUcAAAAJ&hl=en), [Bianca Zadrozny](https://www.linkedin.com/in/biancazadrozny/), [Gabriele Cavallaro](https://www.linkedin.com/in/dr-gabriele-cavallaro/), [Campbell Watson](https://www.linkedin.com/in/campbell-watson-819101100/), [Manil Maskey](https://www.linkedin.com/in/manilmaskey/), [Rahul Ramachandran](https://www.linkedin.com/in/rramachandran05/), [Juan Bernabe Moreno](https://www.linkedin.com/in/bernabemoreno/)  

#### **IBM Research, NASA Marshall Space Flight Center, The University of Alabama in Huntsville, University of Iceland, Jülich Supercomputing Centre, Virginia Tech, Arizona State University, Oregon State University, Clark University, Boston University, University of California, Berkeley, Earth from Space Institute **

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://huggingface.co/ibm-nasa-geospatial)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

This repository contains code and examples based on the [TerraTorch](github.com/IBM/terratorch) library for fine-tuning [Prithvi-EO-2.0](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo), a more powerful version of the foundation model [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) developed by IBM and NASA. Trained on 4.2M global time series samples on the JUWELS HPC system at the Jülich Supercomputing Centre (JSC) using NASA’s Harmonized Landsat and Sentinel data at 30m resolution, it offers significant improvements over its predecessor. 

## Architecture Overview

Prithvi-EO-2.0 is based on the ViT architecture, pre-trained using a masked autoencoder (MAE) approach, with two major modifications as shown in the figure below. First, we introduce a random dropout mechanism that completely removes different bands before the patch embeddings, with the aim of improving the ability of the model to deal with missingness of data. Second, we make modifications to support inputs with temporal and multi-spectral characteristics. 
![model_architecture](https://github.com/user-attachments/assets/d9d52807-f7ca-48bc-b010-e5178f790155)

Our main modifications to the ViT architecture are the 3D positional embedding and the 3D patch embedding, which are required to deal with spatiotemporal data. We have also included metadata and process metadata about the actual geolocation (e.g. latitude and longitude) and date (i.e. year and day-of-year ranging 1-365). This is done by adding biases that are calculated via 2D sine-cosine positional encoding and added to the 3D positional embeddings and 3D patch embeddings via a learned weighted sum (i.e. the weight given is a parameter learned during pretraining). Since this metadata is often not available, we pretrained Prithvi-EO-2.0 allowing for this to be absent via a dropout.

## Pre-trained Models

| Model | Details | Weights |
| ------------- | ------------- | ------------- |
|Prithvi-EO-2.0-300M   | Pretrained 300M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)  |
|Prithvi-EO-2.0-300M-TL   | Pretrained 300M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)  |
|Prithvi-EO-2.0-600M   | Pretrained 600M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) |
|Prithvi-EO-2.0-600M-TL   | Pretrained 600M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)   |


## Benchmarking

We used the most popular and rigorous benchmark framework available for Earth Observation foundation models: [GEO-Bench](https://github.com/ServiceNow/geo-bench). 

## Fine-tuning

We have fined-tuned Prithvi-EO-2.0 for downstream tasks in different domains of interest using [TerraTorch](github.com/IBM/terratorch). Below we provide a list of the downstream tasks, along with links to the datasets, sample TerraTorch configuration files (or custom code, in the case of Gross Primary Product) and sample notebooks for fine-tuning.

| Task | Dataset | TerraTorch Config/Code | Sample Notebook| 
| ------------- | ------------- | ------------- |------------- |
|Flood Detection|[https://github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)| | |
|Wildfire Scar Detection| [https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)| | |
|Burn Scar Intensity| [https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity](https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity)|[test_burnintensity.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/test_burnintensity.yaml)| |
|Landslide Detection|[https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense) | [test_landslide.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/test_landslide.yaml)|[example_landslide4sense.ipynb](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_landslide4sense.ipynb) |
|Multi-temporal Crop Segmentation (US)| [https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)| | |
|Multi-temporal Land Cover and Crop Classification (Europe)|[https://datapub.fz-juelich.de/sen4map/](https://datapub.fz-juelich.de/sen4map/) | | |
|Above Ground Biomass Estimation| [https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters](https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters)|[test_biomassters.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/test_biomassters.yaml) | |
|Gross Primary Productivity Estimation|[https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux)| | |
