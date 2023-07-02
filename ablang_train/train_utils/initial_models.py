from distutils.util import strtobool

import pytorch_lightning as pl


class AbLangPaired_v1(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AbLangPaired")
        
        parser.add_argument('--n_encoder_blocks', type=int, default=1, help='Number of encoder blocks.')
        parser.add_argument('--hidden_embed_size', type=int, default=768, help='Representation (hidden) size.')
        parser.add_argument('--n_attn_heads', type=int, default=12)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--use_tkn_dropout', type=strtobool, default=False)
        parser.add_argument('--loss_fn', type=str, default="CrossEntropy_Loss")
        parser.add_argument('--a_fn', type=str, default="gelu")
        parser.add_argument('--fl_gamma', type=int, default=2)
        parser.add_argument('--use_moe', type=strtobool, default=False)
        
        parser.add_argument('--mask_percent', type=float, default=.15, help='Percentage to mask.')
        parser.add_argument('--variable_masking', type=strtobool, default=False, help='Random uniform masking between 0 and mask_percent for each batch.')
        parser.add_argument('--mask_technique', type=str, default="random", help='masking technique to use. {standard, standard-sl_mask}')
        parser.add_argument('--change_percent', type=float, default=.1, help='Change percent.')
        parser.add_argument('--leave_percent', type=float, default=.1, help='Leave percent.')
        
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--layer_norm_eps', type=float, default=1e-12)

        #max_grad_norm: 1
        #hparams.sync_batchnorm=True # Slower training, but might improve training
        
        return parent_parser
    
    
    @staticmethod
    def add_training_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AbLangPaired")
        parser.add_argument("--data_path", type=str, default='../data/feb2022_5_data')
        parser.add_argument('--out_path', type=str, default="/data/iraqbabbler/olsen/Documents/projects/AbLang/model-catalogue/paired-ablang/train_ablang_pair/reports/models")
        parser.add_argument('--eval_path', type=str, default="/vols/bitbucket/olsen/processed_oas_data/nov2022/nov2022-paired-all/")
        
        parser.add_argument('--cpus', type=int, default=1, help='Number of cpus to use on data handling (4xGPUs is the recommended). \
                                                                    0 uses the main process to load the data.')

        parser.add_argument('--max_fit_batch_size', type=int, default=256, help='Max batch size that fits in GPU memory.')
        parser.add_argument('--effective_batch_size', type=int, default=4_096*2, help='Effective batch size - The real batch size.')
        parser.add_argument('--num_training_steps', type=int, default=1000, help='Number of training steps.')
        parser.add_argument('--warmup_steps', type=int, default=2000, help='Number of warm-up steps.')
        
        parser.add_argument('--learning_rate', type=float, default=2e-04, help='Learning rate')
        parser.add_argument('--cdr3_focus', type=float, default=1, help='Used to increase the chance of masking the CDR3 region. \
                                                                        1 is same as other residues, \
                                                                        2 is 2 times the chance, \
                                                                        3 is 3 times the chance, etc..')        
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam Epsilon.')
        parser.add_argument('--adam_betas', default=[0.9,0.98])
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--eval_batch_size', type=int, default=100)
        parser.add_argument('--over_sample_data', type=int, default=0)
        
        return parent_parser
    
    @staticmethod
    def add_pl_train_args(parent_parser):
        parser = parent_parser.add_argument_group("AbLangPaired")
        
        parser.add_argument('--accelerator', type=str, default="cpu")
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--precision', type=str, default='32-true')       
        parser.add_argument('--val_check_interval', type=int, default=100)
        parser.add_argument('--log_every_n_steps', type=int, default=100)
        parser.add_argument('--enable_checkpointing', type=strtobool, default=None)
        parser.add_argument('--default_root_dir', type=strtobool, default=None)      
        
        return parent_parser