"""
main_flux_finetune_baselines.py
This script reads in HLS S30/L30 data, MERRA-2 processed data uses 
Prithvi 300M/ Prithvi 300 M with temporal and location embeddding to 
regress GPP flux at eddy covariance observation sites.
Author: Srija Chakraborty, Besart Mujeci, MERRA variables selection, flux inputs from Yanghui Kang
Usage:
    python main_flux_finetune_baselines.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
import glob
#from Prithvi_global_v1.mae.models_mae import MaskedAutoencoderViT
import pickle
from Prithvi_global_v1.mae.config import get_config
import argparse
from typing import Optional
from functools import partial
import pandas as pd
import rasterio
from flux_load_data_vInf3 import flux_dataset, flux_dataloader
from flux_regress import RegressionModel_flux
from flux_regress import prithvi
#import wandb
# from Prithvi_global_v1.mae.config import get_config
from PIL import Image
from flux_load_data_vInf3 import load_raster
from flux_load_data_vInf3 import preprocess_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.trainers import BaseTask

from terratorch.models import EncoderDecoderFactory 
from terratorch.datasets import HLSBands
from terratorch.tasks import PixelwiseRegressionTask

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#eval_metrics to log while training
def get_mse(pred, targ):
    criterion = nn.MSELoss()
    mse_loss = criterion(pred, targ)
    return mse_loss.item()

def get_mae(pred, targ):
    criterion = nn.L1Loss()
    mae_loss = criterion(pred, targ)
    return mae_loss.item()

#can use sklearn R2 instead -- as used to evalute whole dataset
def get_r_sq(pred,targ):
    metric = R2Score()
    metric.update(pred, targ)
    r2=metric.compute()
    # Extracting the first element if it's a tensor with multiple elements
    if isinstance(r2, torch.Tensor) and r2.numel() > 1:
        return r2[0].item()
    else:
        return r2.item()

#prithvi encoder for HLS transformation to embedded 1024D space
class prithvi_terratorch(nn.Module):

    def __init__(self, prithvi_weight, model_instance, input_size):

        super(prithvi_terratorch, self).__init__()

        # load checkpoint for Prithvi_global

        self.weights_path = prithvi_weight
        self.checkpoint = torch.load(self.weights_path)
        self.input_size = input_size

        self.prithvi_model = model_instance
        self.prithvi_model.eval() # to double check the prithvi weights being frozen

        _ = self.prithvi_model.load_state_dict(self.checkpoint, strict=False)


    def forward(self,x,temp,loc,mask):
        latent,_,ids_restore, _ = self.prithvi_model.forward_encoder(x,temp,loc,mask) 

        pred = self.prithvi_model.forward_decoder(latent, ids_restore, temp, loc, self.input_size)

        return latent, pred


def main():
    with open('fluxconfig_trainer.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #read model config setups, data mean, std
    device=config["device_name"]
    n_channel=config["model"]["n_channel"]
    n_frame=config["data"]["n_frame"]
    n_iteration=config["n_iteration"]
    checkpoint_dir=config["logging"]["checkpoint_dir"] 
    metrics_dir=config["logging"]["metrics_dir"]
    plots_dir=config["logging"]["plots_dir"]
    train_batch_size=config["training"]["train_batch_size"]
    test_batch_size=config["testing"]["test_batch_size"]
    optim_name=config["training"]["optimizer"]["name"]
    learning_rate=float(config["training"]["optimizer"]["params"]["lr"])
    sch=config["training"]["scheduler"]["use"]
    drp=sch=config["training"]["dropout"]["use"]
    drp_rate=float(config["training"]["dropout"]["val"])
    bn=sch=float(config["training"]["bn"])
    #class_weights=config["class_weights"]
    input_size=config["data"]["input_size"]
    chips=config["data"]["chips"]
    year_to_test=config["test_year"]
    print('TEST YEAR', year_to_test)
    means=config["data"][f"means_for{year_to_test}test"]
    stds=config["data"][f"stds_for{year_to_test}test"]
    merra_means=config["data"][f"merra_means_for{year_to_test}test"]
    merra_stds=config["data"][f"merra_stds_for{year_to_test}test"]
    gpp_means=config["data"][f"gpp_means_for{year_to_test}test"]
    gpp_stds=config["data"][f"gpp_stds_for{year_to_test}test"]
    test_year=config["test_year"]
    norm=config["normalization"]

    #read merra, gpp inputs
    df = pd.read_csv("data_train_hls_37sites_v0_1.csv")

    #get train_test splits
    test_df = df[df['year'] == year_to_test]
    test_chips = test_df['Chip'].tolist()
    merra_test = test_df[['T2MIN', 'T2MAX', 'T2MEAN', 'TSMDEWMEAN', 'GWETROOT', 'LHLAND', 'SHLAND', 'SWLAND', 'PARDFLAND', 'PRECTOTLAND']].values.tolist()
    test_target = test_df['GPP'].tolist()
    
    train_df = df[df['year'] != year_to_test]
    train_chips = train_df['Chip'].tolist()
    merra_train = train_df[['T2MIN', 'T2MAX', 'T2MEAN', 'TSMDEWMEAN', 'GWETROOT', 'LHLAND', 'SHLAND', 'SWLAND', 'PARDFLAND','PRECTOTLAND']].values.tolist()
    train_target = train_df['GPP'].tolist()
    
    means=np.array(means)
    stds=np.array(stds)
    merra_means=np.array(merra_means)
    merra_stds=np.array(merra_stds)
    gpp_means=np.array(gpp_means)
    gpp_stds=np.array(gpp_stds)

    #create ordered paits of hls, merra input and flux output based on splits
    flux_dataset_train=flux_dataset([chips + '/' + str(ele) for ele in train_chips],means,stds, merra_train, merra_means, merra_stds,gpp_means, gpp_stds, train_target)
    flux_dataset_test=flux_dataset([chips + '/' + str(ele) for ele in test_chips],means,stds, merra_test, merra_means, merra_stds,gpp_means, gpp_stds, test_target)

    #data_loader_flux_tr = DataLoader(flux_dataset_train, batch_size=train_batch_size, shuffle=config["training"]["shuffle"])
    #data_loader_flux_test = DataLoader(flux_dataset_test, batch_size=test_batch_size, shuffle=config["testing"]["shuffle"])
    
    datamodule = flux_dataloader(flux_dataset_train, flux_dataset_test, train_batch_size, test_batch_size, config)
    print('done data load')

    wt_file='/dccstor/jlsa931/carbon_flux/checkpoints/checkpoint.pt'#checkpoint for 300M, or 300M T+L

    cf_full=get_config(None)

    #use_model="terratorch_factory" 
    use_model="terratorch"
    #use_model="builtin" 

    if use_model == "builtin":
        prithvi_model = prithvi(prithvi_weight=wt_file, prithvi_config=cf_full,n_frame=1, input_size=[1,50,50])

    elif use_model == "terratorch":
        from terratorch.models.backbones.vit_encoder_decoder import TemporalViTEncoder
        prithvi_instance = TemporalViTEncoder(
                patch_size=cf_full.MODEL.PATCH_SIZE[1],
                num_frames=1,
                in_chans=len(cf_full.DATA.BANDS),
                embed_dim=cf_full.MODEL.EMBED_DIM,
                decoder_depth=8,
                num_heads=cf_full.MODEL.NUM_HEADS,
                mlp_ratio=cf_full.MODEL.MLP_RATIO,
                head_dropout=0.2,
                backbone_input_size=[1,50,50],
                encoder_only=False,
                padding=True,
        )
        prithvi_model = prithvi_terratorch(wt_file, prithvi_instance, [1, 50, 50])

    else:
        model_factory = EncoderDecoderFactory()

        prithvi_model_ = model_factory.build_model(task="regression",
                backbone="prithvi_vit_300",
                decoder="IdentityDecoder",
                backbone_bands=[
                    HLSBands.BLUE,
                    HLSBands.GREEN,
                    HLSBands.RED,
                    HLSBands.NIR_NARROW,
                    HLSBands.SWIR_1,
                    HLSBands.SWIR_2,
                ],
                #num_classes=2,
                backbone_patch_size=cf_full.MODEL.PATCH_SIZE[1],
                backbone_pretrained=False, #True,
                backbone_num_frames=1,
                backbone_in_chans=len(cf_full.DATA.BANDS),
                backbone_embed_dim=cf_full.MODEL.EMBED_DIM,
                backbone_depth=cf_full.MODEL.DEPTH,
                backbone_num_heads=cf_full.MODEL.NUM_HEADS,
                backbone_mlp_ratio=cf_full.MODEL.MLP_RATIO,
                head_dropout=0.2,
                backbone_input_size=[1,50,50],
            )

        prithvi_model = prithvi_model_

    # Instantiate the regression model
    model_comb = RegressionModel_flux(prithvi_model)

    # Set MAE encoder weights to be frozen
    for param in model_comb.prithvi_model.parameters():
        param.requires_grad = False

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model_comb.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    task = PixelwiseRegressionTask(None, None, model=model_comb, loss="mse", optimizer="AdamW")

    accelerator = "cuda"
    checkpoint_callback = ModelCheckpoint(monitor=task.monitor, save_top_k=1, save_last=True)
    num_epochs = n_iteration
    experiment = "carbon_flux"
    default_root_dir = os.path.join("tutorial_experiments", experiment)
    logger = TensorBoardLogger(save_dir=default_root_dir, name=experiment)

    trainer = Trainer(
        # precision="16-mixed",
        accelerator=accelerator,
        callbacks=[
            RichProgressBar(),
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        #logger=logger,
        max_epochs=num_epochs, # train only one epoch for demo
        default_root_dir=default_root_dir,
        log_every_n_steps=1,
        check_val_every_n_epoch=200

    )
    
    trainer.fit(model=task, datamodule=datamodule)

    """
    # Training loop
    num_epochs = n_iteration
    #metrics to track while training
    loss_tot_running=[]
    tot_mae_train=[]
    tot_mse_train=[]
    tot_r2=[]
    best_track_loss = float('inf') #ideally val loss

    for epoch in range(num_epochs):
        running_loss=0.0
        mae_train=[]
        mse_train=[]
        r2_train=[]
        model_comb.train()

        for im2d_batch, pt1d_batch, target_batch in data_loader_flux_tr:

            pt1d_batch=pt1d_batch.to(torch.float32)
            target_batch=target_batch.to(torch.float32)

            # Move the model to the appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_comb.to(device)
            im2d_batch = im2d_batch.to(device)
            pt1d_batch = pt1d_batch.to(device)
            target_batch = target_batch.to(device)


            optimizer.zero_grad()#Zero your gradients for every batch
            # Compute the loss and its gradients
            outputs = model_comb(im2d_batch, pt1d_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            running_loss += loss.item()
    
            #print(f'Epoch {epoch+1}, Loss: {loss.item()}') --per batch loss
            #log metrics for entire batch
            mae_loss=get_mae(outputs, target_batch)#
            mae_train.append(mae_loss)
            mse_loss=get_mse(outputs, target_batch)
            mse_train.append(mse_loss)##loss.item() training on MSE
            r2=get_r_sq(outputs, target_batch)
            r2_train.append(r2)


        #----Reduce LR on plateau
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(loss)#ideally val loss
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: ADAMW lr %.6f -> %.6f" % (epoch, before_lr, after_lr))
        
        #----Reduce LR on plateau
        epoch_loss_train_running=(running_loss)/len(data_loader_flux_tr)
        print(f'Epoch {epoch+1}, TOTAL Loss: {epoch_loss_train_running}')
        r2_train_ep=(np.sum(np.asarray(r2_train)))/len(data_loader_flux_tr)
        mae_train_ep=(np.sum(np.asarray(mae_train)))/len(data_loader_flux_tr)
        mse_train_ep=(np.sum(np.asarray(mse_train)))/len(data_loader_flux_tr)
        print(f'Epoch {epoch+1}, TOTAL metrics: {r2_train_ep}, {mae_train_ep}, {mse_train_ep}')
        loss_tot_running.append(epoch_loss_train_running)
        tot_mae_train.append(mae_train_ep)
        tot_mse_train.append(mse_train_ep)
        tot_r2.append(r2_train_ep)
        #save best -- with train only, ideally needs val set
        if epoch_loss_train_running < best_track_loss:
            print(f"Validation loss decreased ({best_track_loss:.6f} --> {epoch_loss_train_running:.6f}). Saving model...")
            best_track_loss = epoch_loss_train_running
            # Save checkpoint (best model), add tracking learning rate
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_comb.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_track_loss
            }, f'{checkpoint_dir}model_MSELoss_ep{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.pt')#drp{drp_rate}_v2

    print("Training complete.")

    #log metrics per epoch computed in normalized scale

    loss_tot_running= np.asarray(loss_tot_running)
    tot_mae_train= np.asarray(tot_mae_train)
    tot_mse_train=np.asarray(tot_mse_train)
    tot_r2=np.asarray(tot_r2)
    
    train_log_stack=np.transpose(np.vstack((loss_tot_running, tot_mae_train, tot_mse_train, tot_r2)))
    np.savetxt(f"{metrics_dir}train_log_epoch_MSELoss_{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.csv",train_log_stack,fmt='%10.6f', delimiter=',', newline='\n', header='train_loss, mae_per_epoch, mse_per_epoch, r2')

    #save train loss convergence, add val if available
    plt.plot(loss_tot_running)
    plt.ylabel('mse loss (norm)')
    plt.xlabel('epoch')
    plt.savefig(f'{plots_dir}train_loss_MSELoss_{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.png', dpi=180)
    plt.close()

    ############ ---apply on test set (batches) and derive metrics,plots---###################
    # Testing Phase -- test set data in batches
    model_comb.eval()
    pred_test=[]
    targ_test=[]
    
    with torch.no_grad():
        for samp, (im2d_batch, pt1d_batch, target_batch) in enumerate(data_loader_flux_test):
            pt1d_batch=pt1d_batch.to(torch.float32)
            target_batch=target_batch.to(torch.float32)
            # Move the model to the appropriate device
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #model_comb.to(device)
            im2d_batch = im2d_batch.to(device)
            pt1d_batch = pt1d_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model_comb(im2d_batch, pt1d_batch)
            pred = outputs.cpu().numpy()
            targ=target_batch.cpu().numpy()
            pred_test.append(pred)
            targ_test.append(targ)
    # Concatenate predictions across batches
    pred_test = np.concatenate(pred_test, axis=0)
    targ_test = np.concatenate(targ_test, axis=0)

    r_norm=r2_score(targ_test, pred_test)#takes true, pred --R2 test set normalized scale (same as unnorm)

    #unnormalize data with train mean, std to save predictions, compute metrics and save
    #gpp_means, std as np.array
    mean_gpp = gpp_means.reshape(-1,1,1)  # Mean across height and width, for each channel
    stds_gpp = gpp_stds.reshape(-1,1,1)   

    pred_final_unnorm=pred_test*stds_gpp + mean_gpp
    targ_final_unnorm=targ_test*stds_gpp + mean_gpp
    pred_final_unnorm=np.reshape(pred_final_unnorm,(pred_final_unnorm.shape[1],1))
    targ_final_unnorm=np.reshape(targ_final_unnorm,(targ_final_unnorm.shape[1],1))
    r2_unnorm=r2_score(targ_final_unnorm, pred_final_unnorm)#true,pred
    mse_unnorm= (targ_final_unnorm - pred_final_unnorm) ** 2
    mae_unnorm= (np.abs(targ_final_unnorm - pred_final_unnorm)) 
    rel_err_unnorm= (np.abs(targ_final_unnorm-pred_final_unnorm)/targ_final_unnorm)#(obs-exp)/obs -- obs: true reading, exp: model pred
    test_stack=np.hstack((pred_test, targ_test, pred_final_unnorm, targ_final_unnorm, mse_unnorm, mae_unnorm, rel_err_unnorm))
    np.savetxt(f"{metrics_dir}test_eval_MSELoss_ep{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.csv",test_stack,fmt='%10.6f', delimiter=',', newline='\n', header='pred_n, tar_n, pred, tar, mse,mae, rel_err')

    #save r2 figure
    plt.scatter(targ_final_unnorm, pred_final_unnorm, alpha=0.6)
    plt.plot([min(targ_final_unnorm), max(targ_final_unnorm)], [min(targ_final_unnorm), max(targ_final_unnorm)], color='red', lw=2, label='Perfect fit')
    plt.xlabel('True GPP', fontsize=14)
    plt.ylabel('Predicted GPP', fontsize=14)
    plt.grid(True)
    plt.title('R2: ' + str(r2_unnorm))
    plt.savefig(f'{plots_dir}R2_test_MSELoss_{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.png', dpi=180)
    plt.close()

    ############ ---apply on train set (batches) in eval mode and derive metrics,plots---###################
    # evaluate performance on training data
    model_comb.eval()
    pred_test_tr=[]
    targ_test_tr=[]
    
    with torch.no_grad():
        for samp, (im2d_batch, pt1d_batch, target_batch) in enumerate(data_loader_flux_tr):
            pt1d_batch=pt1d_batch.to(torch.float32)
            target_batch=target_batch.to(torch.float32)
            # Move the model to the appropriate device
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #model_comb.to(device)
            im2d_batch = im2d_batch.to(device)
            pt1d_batch = pt1d_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model_comb(im2d_batch, pt1d_batch)
            
            pred = outputs.cpu().numpy()
            targ=target_batch.cpu().numpy()
            pred_test_tr.append(pred)
            targ_test_tr.append(targ)
            
    # Concatenate predictions across batches
    pred_test_tr = np.concatenate(pred_test_tr, axis=0)
    targ_test_tr = np.concatenate(targ_test_tr, axis=0)
    
    #unnormalize and save pred, metrics on full training set
    pred_final_unnorm_tr=pred_test_tr*stds_gpp + mean_gpp
    targ_final_unnorm_tr=targ_test_tr*stds_gpp + mean_gpp
    print(pred_final_unnorm_tr.shape, targ_final_unnorm_tr.shape)
    pred_final_unnorm_tr=np.reshape(pred_final_unnorm_tr,(pred_final_unnorm_tr.shape[1],1))
    targ_final_unnorm_tr=np.reshape(targ_final_unnorm_tr,(targ_final_unnorm_tr.shape[1],1))
    r2_unnorm_tr=r2_score(targ_final_unnorm_tr, pred_final_unnorm_tr)#true,pred
    print(r2_unnorm_tr)
    #mse_norm= (test_pred - test_target) ** 2
    mse_unnorm_tr= (targ_final_unnorm_tr - pred_final_unnorm_tr) ** 2
    mae_unnorm_tr= (np.abs(targ_final_unnorm_tr - pred_final_unnorm_tr)) 
    rel_err_unnorm_tr= (np.abs(targ_final_unnorm_tr-pred_final_unnorm_tr)/targ_final_unnorm_tr)
    train_stack=np.hstack((pred_test_tr, targ_test_tr, pred_final_unnorm_tr, targ_final_unnorm_tr ,mse_unnorm_tr, mae_unnorm_tr, rel_err_unnorm_tr))
    np.savetxt(f"{metrics_dir}train_eval_MSELoss_ep{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.csv",test_stack,fmt='%10.6f', delimiter=',', newline='\n', header='pred_n, tar_n, pred, tar, mse,mae, rel_err')

    #save r2 image
    plt.scatter(targ_final_unnorm_tr, pred_final_unnorm_tr, alpha=0.6)
    plt.plot([min(targ_final_unnorm_tr), max(targ_final_unnorm_tr)], [min(targ_final_unnorm_tr), max(targ_final_unnorm_tr)], color='red', lw=2, label='Perfect fit')
    plt.xlabel('True GPP', fontsize=14)
    plt.ylabel('Predicted GPP', fontsize=14)
    plt.grid(True)
    plt.title('R2: ' + str(r2_unnorm_tr))
    plt.savefig(f'{plots_dir}R2_train_MSELoss_{num_epochs}_lr{learning_rate}_{optim_name}_sc{sch}_yr{year_to_test}.png', dpi=180)
    plt.close()
    """ 


if __name__ == "__main__":
    main()
