import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger

from ablang_train import ABtokenizer, AbLang, trainingframe
from ablang_train.train_utils import callback_handler, datamodule, arghandler

    
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
    
def set_neptune_logger(args):
        """
        Initialize Neptune logger
        """

        neptune_args = { 'api_key':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0N2Y2YmIxMS02OWM3LTRhY2MtYTQxOC0xODU5N2E0ODFmMzEifQ==",
        'project':"tobiasheol/AbLangTraining",
        'name':args.name,
        'log_model_checkpoints':False,
        }
        
        return NeptuneLogger(**neptune_args)


if __name__ == '__main__':
    
    
    # SET ARGUMENTS AND HPARAMS
    arguments = arghandler.parse_args()
    arguments.trainer_args['logger'] = set_neptune_logger(arguments.model_specific_args)

    # SET CALLBACKS
    callbacks = callback_handler.CallbackHandler(
        save_step_frequency=arguments.model_specific_args.val_check_interval, 
        progress_refresh_rate=0, 
        outpath=arguments.model_specific_args.out_path
    )
    
    # SET SEED - IMPORTANT FOR MULTIPLE GPUS, OTHERWISE GOOD FOR REPRODUCIBILITY
    enforce_reproducibility(arguments.model_specific_args.seed)
    
    # LOAD AND INITIATE DATA
    abrep_dm = datamodule.MyDataModule(arguments.model_specific_args, ABtokenizer) 
    
    # You are supposed to just be able to add abrep to the fit function, but it doesn't work when using multiple GPUs
    abrep_dm.setup('fit')

    train = abrep_dm.train_dataloader()
    val = abrep_dm.val_dataloader()
    
    # LOAD MODEL
    model = trainingframe.TrainingFrame(arguments.model_specific_args, AbLang, ABtokenizer)

    # INITIALISE TRAINER
    trainer = pl.Trainer(**arguments.trainer_args, callbacks=callbacks())

    # TRAIN MODEL
    trainer.fit(model, train, val)
    
