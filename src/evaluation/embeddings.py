from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from neptune.new.types import File


def plot_Ab_embeddings(trainer):
    
    xembeds = trainer.AbLang.AbRep.AbEmbeddings.AAEmbeddings.weight.detach().cpu().numpy()
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
    
def plot_Ap_embeddings(trainer):
    
    xembeds = trainer.AbLang.AbRep.AbEmbeddings.PositionEmbeddings.weight.detach().cpu().numpy()

    pos_embeds = pos_matrix(xembeds[1:])
    np.fill_diagonal(pos_embeds, 0)

    fig, ax = plt.subplots(figsize=(9,6))

    sns.heatmap(pos_embeds, cmap="coolwarm", ax=ax, cbar=False, xticklabels=10, vmin=-.2, vmax=.7)
    fig.colorbar(ax.collections[0], ax=ax, location="right", use_gridspec=False, pad=0.2)
    
    trainer.logger.experiment["evaluation/pos_embeddings"].log(File.as_image(fig))
    plt.close()
    
def pos_matrix(pos_embeds):
    similarity = np.dot(pos_embeds, pos_embeds.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    
    return cosine