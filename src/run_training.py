import time, torch
import numpy as np
import pytorch_lightning as pl
from ablang_pair import model, tokenizers

from custom_callbacks.callback_handler import CallbackHandler
import trainingframe, arghandler
from data_handling import datamodule

    
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
    trainer_args, hparams = arghandler.parse_args()
    
    # SET CALLBACKS
    callbacks = CallbackHandler(n_steps=100, progress_refresh_rate=0, outpath=hparams.outpath)
    
    # SET SEED - IMPORTANT FOR MULTIPLE GPUS, OTHERWISE GOOD FOR REPRODUCIBILITY
    enforce_reproducibility(hparams.seed)
    
    # LOAD AND INITIATE DATA
    abrep_dm = datamodule.MyDataModule(hparams, tokenizers) 
    # You are supposed to just be able to add abrep to the fit function, but it doesn't work when using multiple GPUs
    abrep_dm.setup('fit')
    
    train = abrep_dm.train_dataloader()
    val = abrep_dm.val_dataloader()
    
    # LOAD MODEL
    model = trainingframe.TrainingFrame(hparams, model, tokenizers)
    
    # INITIALISE TRAINER
    trainer = pl.Trainer(**trainer_args, callbacks=callbacks())
    
    start_time = time.time()
    # TRAIN MODEL
    trainer.fit(model, train, val)
    
    print(time.time()-start_time)
    
