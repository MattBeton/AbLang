from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from neptune.new.types import File


def plot_Ab_embeddings(trainer):
    
    xembeds = trainer.ablang.get_aa_embeddings.weight.detach().cpu().numpy()
    vocabs = trainer.tokenizer.vocab_to_aa
    
    
    desired_tokens = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23]
    xembeds = xembeds[desired_tokens]
    vocabs = [vocabs[key] for key in desired_tokens]

    result = PCA(n_components=2, svd_solver='full').fit_transform(xembeds)
    
    simple_vocab = {'R':'Charged basic','H':'Charged basic','K':'Charged basic',
                    'D':'Charged acidic','E':'Charged acidic',
                    'S':'Polar neutral','T':'Polar neutral','N':'Polar neutral','Q':'Polar neutral','C':'Polar neutral',
                    'G':'Unique','P':'Unique','U':'Unique','O':'Unique',
                    'A':'Hydrophobic aliphatic','V':'Hydrophobic aliphatic','I':'Hydrophobic aliphatic',
                    'L':'Hydrophobic aliphatic','M':'Hydrophobic aliphatic',
                    'F':'Hydrophobic aromatic','Y':'Hydrophobic aromatic','W':'Hydrophobic aromatic',
                    '<':'Token','-':'Token','>':'Token','*':'Token'}
    
    simple_vocabs = [simple_vocab[char] for char in vocabs]
    
    g = sns.relplot(x=result[:,0], y=result[:,1], hue=simple_vocabs, kind="scatter", palette="Set1", s=100, height=7, aspect=1.3)

    for ax in g.axes.flat:
        for (x, y), val in zip(result, vocabs):
            ax.text(x-0.02, y+0.007, val, fontsize=15)
            
    trainer.logger.experiment["evaluation/aa_embeddings"].log(File.as_image(g.fig)) #.upload(g.fig)
    plt.close()