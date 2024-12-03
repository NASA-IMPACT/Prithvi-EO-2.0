# CO_2 Flux:

This is a regression task where HLS images and corresponding MERRA-2 data for a region are passed in parallel
to the model and combine the result to predict CO_2 flux value for that region. 

HLS images are passed through prithvi_pretrained model, but not the MERRA-2 data.

However both prithvi output and MERRA-2 data are projected in same embedding space to combine.

# Data 

Available from https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux

To download the dataset, you can use the script `download_dataset.py` as seen below:
```
python download_dataset.py --save_dir <directory to save the dataset>
```
It will create two subdirectories called `train` and `test` inside the download directory. 

After that, you can adapt the config file `fluxconfig_trainer.yaml` correspondig to the paths for the datasets:

```yaml
data:
  n_frame: 1
  chips: "/dccstor/jlsa931/carbon_flux/train/images/" # Example directory
  test_chips: "/dccstor/jlsa931/carbon_flux/test/images/" #Example diretory
  input_size: [6,50, 50]
  means_for2018test: [0.07286696773903256, 0.10036772476940378, 0.11363777043869523, 0.2720510638470194, 0.2201167122609674, 0.1484162876040495]
  stds_for2018test: [0.13271414936598172, 0.13268933338964875, 0.1384673725283858, 0.12089142598551804, 0.10977084890500641, 0.0978705241034744]
...

```

you can run the notebook using:

```
jupyter lab  main_flux_finetune_baselines_trainer.ipynb
```

![Screenshot 2024-11-05 at 4 02 20 PM](https://github.com/user-attachments/assets/033a0b1f-328f-430f-9b0f-72f64ba7321c)

