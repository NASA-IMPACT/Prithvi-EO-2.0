# Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Prithvi-EO-2.0">
</p>

#### [Daniela Szwarcman](https://www.linkedin.com/in/daniela-szwarcman-60b55876/), [Sujit Roy](https://www.linkedin.com/in/sujit-roy01/), [Paolo Fraccaro](https://www.linkedin.com/in/paolo-fraccaro-3b85371b/?originalSubdomain=uk), [Þorsteinn Elí Gíslason](https://www.linkedin.com/in/%C3%BEorsteinn-el%C3%AD-g%C3%ADslason-a6ab951a9), [Benedikt Blumenstiel](https://www.linkedin.com/in/blumenstiel/), [Rinki Ghosal](https://www.linkedin.com/in/rinki-ghosal-5b2a41106/), [Pedro Henrique de Oliveira](https://www.linkedin.com/in/pedro-henrique-conrado-ferreira-de-oliveira-420377220/), [João Lucas de Sousa Almeida](https://www.linkedin.com/in/jo%C3%A3o-lucas-de-sousa-almeida-a08b9255/), [Rocco Sedona](https://www.linkedin.com/in/rocco-sedona-79812749/), [Yanghui Kang](https://www.linkedin.com/in/yanghui-kang-797aa33a/), [Srija Chakraborty](https://www.linkedin.com/in/chakrabortysrija/), [Sizhe Wang](https://scholar.google.com/citations?user=bucEAU0AAAAJ&hl=en), [Ankur Kumar](https://www.linkedin.com/in/ankurk017/), [Myscon Truong](https://www.linkedin.com/in/myscon-truong/), [Denys Godwin](https://www.linkedin.com/in/denys-godwin-43a49188/), [Hyunho Lee](https://scholar.google.com/citations?user=oOwJeyQAAAAJ), [Chia-Yu Hsu](https://www.linkedin.com/in/chiayu-hsu/), [Ata Akbari Asanjan](https://www.linkedin.com/in/ataakbariasanjan/), [Besart Mujeci](https://www.linkedin.com/in/besart/), [Trevor Keenan](https://www.linkedin.com/in/trevor-keenan/), [Paulo Arévolo](https://scholar.google.com/citations?user=AwYBme4AAAAJ&hl=en), [Wenwen Li](https://www.linkedin.com/in/wenwenli/), [Hamed Alemohammad](https://www.linkedin.com/in/hamedalemo/), [Pontus Olofsson](https://www.linkedin.com/in/pontus-olofsson-057701255/), [Christopher Hain](https://www.linkedin.com/in/christopher-hain-5b465917b/), [Robert Kennedy](https://scholar.google.com/citations?user=I-2_GUcAAAAJ&hl=en), [Bianca Zadrozny](https://www.linkedin.com/in/biancazadrozny/), [Gabriele Cavallaro](https://www.linkedin.com/in/dr-gabriele-cavallaro/), [Campbell Watson](https://www.linkedin.com/in/campbell-watson-819101100/), [Manil Maskey](https://www.linkedin.com/in/manilmaskey/), [Rahul Ramachandran](https://www.linkedin.com/in/rramachandran05/), [Juan Bernabe Moreno](https://www.linkedin.com/in/bernabemoreno/)  

#### **IBM Research, NASA Marshall Space Flight Center, The University of Alabama in Huntsville, University of Iceland, Jülich Supercomputing Centre, Virginia Tech, Arizona State University, Oregon State University, Clark University, Boston University, University of California, Berkeley, Earth from Space Institute **

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://huggingface.co/ibm-nasa-geospatial)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.02732)

This repository contains code and examples based on the [TerraTorch](https://github.com/IBM/terratorch) library for fine-tuning [Prithvi-EO-2.0](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo), a more powerful version of the foundation model [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) developed by IBM and NASA. Trained on 4.2M global time series samples on the JUWELS HPC system at the Jülich Supercomputing Centre (JSC) using NASA’s Harmonized Landsat and Sentinel data at 30m resolution, it offers significant improvements over its predecessor. 

## 📢 Latest Updates

- **December 4, 2024**: Prithvi-EO-2.0 pre-trained models and fine-tuning datasets released on [Hugging Face](https://huggingface.co/ibm-nasa-geospatial). 
- **December 5, 2024**: Prithvi-EO-2.0 paper released on [arxiv link](https://arxiv.org/abs/2412.02732). 🔥🔥

## Architecture Overview

Prithvi-EO-2.0 is based on the ViT architecture, pretrained using a masked autoencoder (MAE) approach, with two major modifications as shown in the figure below. 

![model_architecture_v2](https://github.com/user-attachments/assets/378c4d18-9a4f-4a9e-bd72-925fb9ed1b41)

First, we replaced the 2D patch embeddings and 2D positional embeddings with 3D versions to support inputs with spatiotemporal characteristics, i.e., a sequence of `T` images of size `(H, W)`. Our 3D patch embeddings consist of a 3D convolutional layer, dividing the 3D input into non-overlapping cubes of size `(t, h, w)` for time, height, and width dimensions, respectively. For the 3D positional encodings, we first generate 1D sin/cos encodings individually for each dimension and then combine them together into a single, 3D positional encoding.

Second, we considered geolocation (center latitude and longitude) and date of acquisition (year and day-of-year ranging 1-365) in pretraining. Both encoder and decoder receive time and location information for each sample and encodes them independently using 2D sin/cos encoding. They are added to the embedded tokens via a weighted sum with learned weights: one for time and one for location and separate weights for encoder and decoder. Since this metadata is often not available, we added a drop mechanism during pretraining that randomly drops the geolocation and/or the temporal data to help the model learn how to handle the absence of this information.




## Pre-trained Models

| Model | Details | Weights |
| ------------- | ------------- | ------------- |
|Prithvi-EO-2.0-300M   | Pretrained 300M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)  |
|Prithvi-EO-2.0-300M-TL   | Pretrained 300M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)  |
|Prithvi-EO-2.0-600M   | Pretrained 600M parameter model  | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) |
|Prithvi-EO-2.0-600M-TL   | Pretrained 600M parameter model with temporal and location embeddings | [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)   |


## Benchmarking

We validated the Prithvi-EO-2.0 models through extensive experiments using [GEO-Bench](https://github.com/ServiceNow/geo-bench), the most popular and rigorous benchmark framework available for Earth Observation foundation models. Prithvi-EO-2.0-600M-TL outperforms the previous Prithvi-EO model by 8\% across a range of tasks. It also outperforms six other geospatial foundation models when benchmarked on remote sensing tasks from different domains and resolutions (i.e. from 0.1m to 15m). 

<img src="https://github.com/user-attachments/assets/b7e49289-810c-4bbc-b127-a361427a259a" width="750" height="450">

## Fine-tuning

We have fined-tuned Prithvi-EO-2.0 for downstream tasks in different domains of interest using [TerraTorch](https://github.com/IBM/terratorch) (see instructions on how to get started [here](https://github.com/IBM/terratorch?tab=readme-ov-file#pip)). Below we provide a list of the downstream tasks, along with links to the datasets, sample TerraTorch configuration files (or custom code, in the case of Gross Primary Product) and sample notebooks for fine-tuning.

### Sample configs

| Task | Dataset | TerraTorch Config/Code | 
| ------------- | ------------- | ------------- |
|Flood Detection|[https://github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)|[sen1floods11.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen1floods11.yaml) | 
|Wildfire Scar Detection| [https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)| [firescars.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/firescars.yaml)|  
|Burn Scar Intensity| [https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity](https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity)|[burnintensity.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/burnintensity.yaml)| 
|Landslide Detection|[https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense) | [landslide.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/landslide.yaml)|
|Multi-temporal Crop Segmentation (US)| [https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)| [multicrop.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/multicrop.yaml)|
|Multi-temporal Land Cover and Crop Classification (Europe)|[https://datapub.fz-juelich.de/sen4map/](https://datapub.fz-juelich.de/sen4map/) | [sen4map_land-cover.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen4map_land-cover.yaml)  [sen4map_crops.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen4map_crops.yaml)
|Above Ground Biomass Estimation| [https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters](https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters)|[biomassters.yaml](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/biomassters.yaml) |
<!--- |Gross Primary Productivity Estimation|[https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux)|[carbon_flux](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/tree/main/examples/carbon_flux)| ---> 

### Sample Fine-tuning Notebooks

* [Landslide Detection](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_landslide4sense.ipynb) 

* [Multi-temporal Crop Segmentation (US)](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_multitemporalcrop.ipynb)

<!--- * [Gross Primary Productivity Estimation](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/refactory/examples/carbon_flux/main_flux_finetune_baselines_trainer.ipynb) --->
