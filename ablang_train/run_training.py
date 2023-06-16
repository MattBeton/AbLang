import os
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from ablang_train import ABtokenizer, AbLang, TrainingFrame, CallbackHandler, AbDataModule, ablang_parse_args

    
def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    #
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    np.random.seed(seed)
    
    pl.seed_everything(seed)

if __name__ == '__main__':
    
    
    # SET ARGUMENTS AND HPARAMS
    arguments = ablang_parse_args()

    # SET CALLBACKS
    callbacks = CallbackHandler(
        save_step_frequency=arguments.model_specific_args.log_every_n_steps, 
        progress_refresh_rate=0, 
        outpath=arguments.model_specific_args.out_path
    )
    
    # SET SEED - IMPORTANT FOR MULTIPLE GPUS, OTHERWISE GOOD FOR REPRODUCIBILITY
    enforce_reproducibility(arguments.model_specific_args.seed)
    
    # LOAD AND INITIATE DATA
    ablang_dm = AbDataModule(arguments.model_specific_args, ABtokenizer) 
    
    # You are supposed to just be able to add abrep to the fit function, but it doesn't work when using multiple GPUs
    ablang_dm.setup('fit')

    train = ablang_dm.train_dataloader()
    val = ablang_dm.val_dataloader()
    
    # LOAD MODEL
    model = TrainingFrame(arguments.model_specific_args, AbLang, ABtokenizer)

    # INITIALISE TRAINER
    trainer = pl.Trainer(**arguments.trainer_args, callbacks=callbacks())

    # TRAIN MODEL
    trainer.fit(model, train, val)
    
