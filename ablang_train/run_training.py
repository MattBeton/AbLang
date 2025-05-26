import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from ablang_train import ABtokenizer, AbLang, TrainingFrame, CallbackHandler, AbDataModule, ablang_parse_args
from pytorch_lightning.callbacks import OnExceptionCheckpoint

    
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
    torch.set_float32_matmul_precision('high')
    
    # SET ARGUMENTS AND HPARAMS
    arguments = ablang_parse_args()
    # print(arguments)

    # If manual optimization is used, Trainer's accumulate_grad_batches should be 1 (or None/default)
    # as accumulation is handled inside the LightningModule's training_step.
    # The model_specific_args.accumulate_grad_batches will still be used by the manual logic.
    if hasattr(arguments.model_specific_args, 'manual_optimization') and arguments.model_specific_args.manual_optimization or True:
        arguments.trainer_args['accumulate_grad_batches'] = 1 
    elif 'accumulate_grad_batches' in arguments.trainer_args and arguments.trainer_args.get('accelerator') == 'mps': # Temp fix for MPS issue with auto accum.
        # Check if manual optimization is implied by TrainingFrame setting
        # This part is tricky as TrainingFrame.automatic_optimization is set instance-level
        # A cleaner way would be to have a hparam that explicitly states manual optimization is on.
        # For now, assuming if we reach here, manual opt is active due to prior changes.
        arguments.trainer_args['accumulate_grad_batches'] = 1

    # SET CALLBACKS
    callbacks = CallbackHandler(
        save_step_frequency=arguments.model_specific_args.log_every_n_steps, 
        progress_refresh_rate=0, 
        outpath=arguments.model_specific_args.out_path
    )
    
    # Add exception checkpoint callback for saving on Ctrl+C
    exception_callback = OnExceptionCheckpoint(
        dirpath=arguments.model_specific_args.out_path,
        filename="interrupted_checkpoint"
    )
    all_callbacks = callbacks() + [exception_callback]
    
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
    print(arguments.trainer_args)
    trainer = pl.Trainer(**arguments.trainer_args, callbacks=all_callbacks)

    # TRAIN MODEL
    trainer.fit(model, train, val)
    
