import torch


heavy_seq = '<EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS>'
light_seq = '<DIVMTQTPSTLSASVGDRVTLTCKASQDISYLAWYQQKPGKAPKKLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQQNSNWTFGQGTKVDIK>'



def log_restoring_sequence(trainer):

        aaPreds, aaPreds50 = restore_seq(trainer, seq_to_restore = heavy_seq+'|')       
        trainer.logger.experiment['evaluation/heavy_reconstruct'].log(str(aaPreds[0]))
        trainer.logger.experiment['evaluation/heavy_reconstruct_50'].log(str(aaPreds50[0]))
        
        aaPreds, aaPreds50 = restore_seq(trainer, seq_to_restore = '|'+light_seq)       
        trainer.logger.experiment['evaluation/light_reconstruct'].log(str(aaPreds[0]))
        trainer.logger.experiment['evaluation/light_reconstruct_50'].log(str(aaPreds50[0]))

        aaPreds, aaPreds50 = restore_seq(trainer, seq_to_restore = heavy_seq+'|'+light_seq)       
        trainer.logger.experiment['evaluation/paired_reconstruct'].log(str(aaPreds[0]))
        trainer.logger.experiment['evaluation/paired_reconstruct_50'].log(str(aaPreds50[0]))
        
        
def restore_seq(trainer, seq_to_restore):
    """
    Small function used to visualize the training by showing how the reconstruction of a given sequence is improved over training.
    """
    
    model = trainer.ablang
    tokenizer = trainer.tokenizer
    
    with torch.no_grad():
        tokenPreds = model(tokenizer([seq_to_restore], pad=True, add_extra_tkns=False, device=trainer.device))
    
    tokenMAX = torch.max(torch.nn.Softmax(dim=-1)(tokenPreds), -1)

    aaPreds = tokenizer(tokenMAX[1], encode=False)

    unkMatrix = torch.zeros(tokenMAX[0].shape, dtype=torch.long, device=trainer.device) + 21
    
    aaPreds50 = ['-'.join(tokenizer(torch.where(tokenMAX[0]<=.5, unkMatrix, tokenMAX[1]).detach(), encode=False)[0].split('<unk>'))]

    return aaPreds, aaPreds50