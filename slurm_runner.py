# Running multiple experiments, have to be on different GPUs each. Used for SLURM.
import argparse
import os
import subprocess as sp
import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import copy
import wandb
from py_scripts import *
#from exp_commands.temp_slurm_jobs import *
#from exp_commands import *
from py_scripts.combine_params import *
import importlib

args = None  # makes args into a global variable that is set during the experiment run



def train_func():
 
    if args.sweep_id:
        wandb.init(
            project=wandb_project,
            entity="",
            dir=output_directory,
        )
        exp_settings = vars(wandb.config)
        exp_settings = exp_settings["_items"]
        print("Sweep Exp settings", exp_settings)
        exp_settings["save_model_checkpoints"] = False
    else:
        # This is using the list provided above. 
        job_script = importlib.import_module(args.job_script) # "exp_commands."+
        exp_settings = init_exp_settings(args.exp_ind,job_script)

    model, data_module, params, callbacks, checkpoint_callback = compile_experiment(exp_settings, args.num_workers)

    temp_trainer = pl.Trainer(
        #precision=16, 
        logger=params.logger,
        max_epochs=params.epochs_to_train_for,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        gpus=-1,
        auto_select_gpus=True,
        enable_progress_bar=False,
        callbacks=callbacks,
        checkpoint_callback = checkpoint_callback, 
        detect_anomaly=True,
        reload_dataloaders_every_n_epochs=params.epochs_per_cl_task,
    )
    temp_trainer.fit(model, data_module, ckpt_path=params.fit_load_state)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_script", type=str, required=False, help="The job script.")
    parser.add_argument("--exp_ind", type=int, required=True, help="The job index.")
    parser.add_argument(
        "--num_workers", type=int, required=True, help="Num CPUS for the workers."
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        required=True,
        help="Can check the number of tasks equals the number of experiments.",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=False,
        default=None,
        help="Can check the number of tasks equals the number of experiments.",
    )
    args = parser.parse_args()

    if args.sweep_id:  
        # run hyperparameter sweep with settings provided for by the hyper sweep agent as the exp_settings.
        print("Sweep id is:", args.sweep_id)
        wandb.agent(
            args.sweep_id,
            function=train_func,
            project=wandb_project,
            entity="",
        )

    else:  # run all of the experiments defined at the top.
        train_func()

