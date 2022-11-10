import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from neptune.new.types import File


desired_tokens = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23]

def plot_aa_embeddings(trainer):
    """
    PCA of the embedding layer. 
    """
    
    aa_embeds = trainer.ablang.get_aa_embeddings().weight.detach().cpu().numpy()   
    
    aa_embeds = aa_embeds[desired_tokens]
    aa_pca = PCA(n_components=2, svd_solver='full').fit_transform(aa_embeds)
    
    vocabs = trainer.tokenizer.vocab_to_aa
    vocabs = [vocabs[key] for key in desired_tokens]
    vocab_w_description = {'R':'Charged basic','H':'Charged basic','K':'Charged basic',
                    'D':'Charged acidic','E':'Charged acidic',
                    'S':'Polar neutral','T':'Polar neutral','N':'Polar neutral','Q':'Polar neutral','C':'Polar neutral',
                    'G':'Unique','P':'Unique','U':'Unique','O':'Unique',
                    'A':'Hydrophobic aliphatic','V':'Hydrophobic aliphatic','I':'Hydrophobic aliphatic',
                    'L':'Hydrophobic aliphatic','M':'Hydrophobic aliphatic',
                    'F':'Hydrophobic aromatic','Y':'Hydrophobic aromatic','W':'Hydrophobic aromatic',
                    '<':'Token','-':'Token','>':'Token','*':'Token'}
    descriptions = [vocab_w_description[aa] for aa in vocabs]
    
    g = sns.relplot(x=aa_pca[:,0], y=aa_pca[:,1], hue=descriptions, kind="scatter", palette="Set1", s=100, height=7, aspect=1.3)

    for ax in g.axes.flat:
        for (x, y), val in zip(aa_pca, vocabs):
            ax.text(x-0.02, y+0.007, val, fontsize=15)
            
    trainer.logger.experiment["evaluation/aa_embeddings"].log(File.as_image(g.fig))
    plt.close()