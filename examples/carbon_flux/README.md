# CO_2 Flux:

This is a regression task where HLS images and corresponding MERRA-2 data for a region are passed in parallel
to the model and combine the result to predict CO_2 flux value for that region. 

HLS images are passed through prithvi_pretrained model, but not the MERRA-2 data.

However both prithvi output and MERRA-2 data are projected in same embedding space to combine.

# Data 

Available from https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux


Run

```
python main_flux_finetune_baselines.py
```
![Screenshot 2024-11-05 at 4 02 20 PM](https://github.com/user-attachments/assets/033a0b1f-328f-430f-9b0f-72f64ba7321c)

