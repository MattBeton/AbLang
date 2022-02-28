import torch

def log_valuation_loss(self, val_step_outputs):


    val_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
    fw1_loss = torch.stack([x['fw1_loss'] for x in val_step_outputs]).mean()
    cdr1_loss = torch.stack([x['cdr1_loss'] for x in val_step_outputs]).mean()
    fw2_loss = torch.stack([x['fw2_loss'] for x in val_step_outputs]).mean()
    cdr2_loss = torch.stack([x['cdr2_loss'] for x in val_step_outputs]).mean()
    fw3_loss = torch.stack([x['fw3_loss'] for x in val_step_outputs]).mean()
    cdr3_loss = torch.stack([x['cdr3_loss'] for x in val_step_outputs]).mean()
    fw4_loss = torch.stack([x['fw4_loss'] for x in val_step_outputs]).mean()

    self.logger.experiment["evaluation/eval_loss"].log(val_loss)
    self.logger.experiment["evaluation/fw1_loss"].log(fw1_loss)
    self.logger.experiment["evaluation/cdr1_loss"].log(cdr1_loss)
    self.logger.experiment["evaluation/fw2_loss"].log(fw2_loss)
    self.logger.experiment["evaluation/cdr2_loss"].log(cdr2_loss)
    self.logger.experiment["evaluation/fw3_loss"].log(fw3_loss)
    self.logger.experiment["evaluation/cdr3_loss"].log(cdr3_loss)
    self.logger.experiment["evaluation/fw4_loss"].log(fw4_loss)

    
def log_restoring_sequence(self):
    testSeq = 'EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS|'
    aaPreds, aaPreds50 = singleSeqValidation(self, self.tokenizer, testSeq=testSeq)       
    self.logger.experiment['evaluation/heavy_reconstruct'].log(aaPreds[0])
    self.logger.experiment['evaluation/heavy_reconstruct_50'].log(aaPreds50[0])

    testSeq = '|DIVMTQTPSTLSASVGDRVTLTCKASQDISYLAWYQQKPGKAPKKLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQQNSNWTFGQGTKVDIK'
    aaPreds, aaPreds50 = singleSeqValidation(self, self.tokenizer, testSeq=testSeq)       
    self.logger.experiment['evaluation/light_reconstruct'].log(aaPreds[0])
    self.logger.experiment['evaluation/light_reconstruct_50'].log(aaPreds50[0])
        
        
def singleSeqValidation(model, tokenizer, testSeq):
    """
    Small function used to visualize the training by showing how the reconstruction of a given sequence is improved over training.
    """

    tokenPreds = model(tokenizer([testSeq], pad=True, device=model.device))
    
    tokenMAX = torch.max(torch.nn.Softmax(dim=-1)(tokenPreds), -1)

    aaPreds = tokenizer(tokenMAX[1], encode=False)

    unkMatrix = torch.zeros(tokenMAX[0].shape, dtype=torch.long, device=model.device) + 21
    
    aaPreds50 = ['-'.join(tokenizer(torch.where(tokenMAX[0]<=.5, unkMatrix, tokenMAX[1]).detach(), encode=False)[0].split('<unk>'))]

    return aaPreds, aaPreds50