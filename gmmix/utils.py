import wandb

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns


from dosnes import dosnes as dosnes_pkg
#import dosnes
#import pdb; pdb.set_trace()

def dosnes(image_embed, text_embed, name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])

    metric = "sqeuclidean"
    momentum = 0.1
    final_momentum = 0.7
    mom_switch_iter = 250
    max_iter = 1000
    learning_rate = 400
    min_gain = 0.01
    model = dosnes_pkg.DOSNES(momentum = momentum, final_momentum = final_momentum, learning_rate = learning_rate, min_gain = min_gain,max_iter = 1000, verbose_freq = 10, metric = metric, verbose = 1, random_state=0)

    len_img = img_np.shape[0]
    X = all_emb
    y = np.concatenate( ( np.ones(shape=(len_img,)) , np.zeros(shape=(len_img,)) ) ,dtype=np.float32)
    
    aaa = model.fit_transform(X, y, filename="training.gif")

    #aaa = TSNE(n_components=3,init='random',n_jobs=1).fit_transform(all_emb)
    np.save(f'{name}_dosnes_vec_{len_img}.npy',aaa)
    plt.clf()
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(aaa[:len_img,0],aaa[:len_img,1],aaa[:len_img,2],c='#FFC514',alpha=0.6,edgecolors='white')
    ax.scatter(aaa[len_img:,0],aaa[len_img:,1],aaa[len_img:,2],c='#00C97A',alpha=0.6,edgecolors='white')
    #plt.legend(["Image","Text"],fontsize="large")
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})
    return plt

def tsne_vec_3d(image_embed,text_embed,name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])
    aaa = TSNE(n_components=3,init='random',n_jobs=1).fit_transform(all_emb)
    #bbb = LocallyLinearEmbedding(n_components=2).fit_transform(all_emb)
    #ccc = PCA(n_components=2).fit_transform(all_emb)
    len_img = img_np.shape[0]

    #plt.clf()
    plt.clf()
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    #plt.show()
    ax.scatter(aaa[:len_img,0],aaa[:len_img,1],aaa[:len_img,2],c='#FFC514',alpha=0.6,edgecolors='white')
    ax.scatter(aaa[len_img:,0],aaa[len_img:,1],aaa[len_img:,2],c='#00C97A',alpha=0.6,edgecolors='white')
    #plt.legend(["Image","Text"],fontsize="large")
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})

    #plt.clf()
    #plt.scatter(bbb[:len_img,0],bbb[:len_img,1])
    #plt.scatter(bbb[len_img:,0],bbb[len_img:,1])
    #wandb.log({"LLE"+name:wandb.Image(plt)})

    #plt.clf()
    #plt.scatter(ccc[:len_img,0],ccc[:len_img,1])
    #plt.scatter(ccc[len_img:,0],ccc[len_img:,1])
    #wandb.log({"PCA"+name:wandb.Image(plt)})
    return plt

def tsne_vec(image_embed,text_embed,name="train"):
    img_np = image_embed.float().detach().cpu().numpy()
    txt_np = text_embed.float().detach().cpu().numpy()
    all_emb = np.concatenate([img_np,txt_np])
    aaa = TSNE(n_components=2,init='random',n_jobs=1).fit_transform(all_emb)
    bbb = LocallyLinearEmbedding(n_components=2).fit_transform(all_emb)
    #ccc = PCA(n_components=2).fit_transform(all_emb)
    len_img = img_np.shape[0]

    #plt.clf()
    plt.clf()
    #plt.show()
    sns.set_style("ticks")
    plt.scatter(aaa[:len_img,0],aaa[:len_img,1],c='#FFC514',alpha=0.6,edgecolors='white')
    plt.scatter(aaa[len_img:,0],aaa[len_img:,1],c='#00C97A',alpha=0.6,edgecolors='white')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.legend(["Image","Text"],fontsize="large")
    plt.savefig("ffig.pdf",pad_inches = 0)
    wandb.log({"tsne"+name:wandb.Image(plt)})

    #plt.clf()
    #plt.scatter(bbb[:len_img,0],bbb[:len_img,1])
    #plt.scatter(bbb[len_img:,0],bbb[len_img:,1])
    #wandb.log({"LLE"+name:wandb.Image(plt)})

    #plt.clf()
    #plt.scatter(ccc[:len_img,0],ccc[:len_img,1])
    #plt.scatter(ccc[len_img:,0],ccc[len_img:,1])
    #wandb.log({"PCA"+name:wandb.Image(plt)})
    return plt
def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def add_key_prefix(dic,pre):
    return {pre+k : v for k,v in dic.items()}
