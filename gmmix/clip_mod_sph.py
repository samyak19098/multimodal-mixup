#!/usr/bin/env python
# coding: utf-8

# In[60]:
import wandb


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--startlr',type=float,default=7e-6,help="initial learning rate")
parser.add_argument('--lr',type=float,default=1e-6,help="initial learning rate")
parser.add_argument('--classifier_lr', type=float,default=1e-6,help="classifier initial learning rate")
parser.add_argument('--bs',type=int,default=16,help="Batch size")
parser.add_argument('--epoch',type=int,default=10,help="Epoch")
parser.add_argument('--group',  help="run Group",default="SWSW")
parser.add_argument('--input_img', help="run Group", action="store_true")
parser.add_argument('--input_cutmix', help="run Group", action="store_true")
parser.add_argument('--manifold_img',  help="run Group",  action="store_true")
parser.add_argument('--out_img',  help="run Group",  action="store_true")
parser.add_argument('--out_txt',  help="run Group",  action="store_true")
parser.add_argument('--original_neg', help="Neg-Neg is Original(Not Mixed)", action="store_true")
parser.add_argument('--manifold_txt',  help="run Group",  action="store_true")
parser.add_argument('--manifold_all',  help="run Group",  action="store_true")
parser.add_argument('--dual_mix',  help="run Group",  action="store_true")
parser.add_argument('--neg_mix',  help="run Group",  action="store_true")
parser.add_argument('--method',type=str, help="method(imix,unmix)")
parser.add_argument('--rmse',  help="run Group",  action="store_true")
parser.add_argument('--reinit',  help="run Group",  action="store_true")
parser.add_argument('--shared',  help="run Group",  action="store_true")
parser.add_argument('--ssnmix',  help="run Group",  action="store_true")
parser.add_argument('--valid',   help="run Group",  action="store_true")
parser.add_argument('--fgsm',   help="run Group",  action="store_true")
parser.add_argument('--weight',   help="load pretained weight",type=str,default=None)
parser.add_argument('--divt',type=float,default=1.0,help="Temperature Dividing Scaler")
parser.add_argument('--perc', type=float,default=1.0,help="Data Percentage")

parser.add_argument('--dusoft',  help="Dual softmax At Inference.",  action="store_true")

parser.add_argument('--dataset',type=str,default="flickr",help="Dataset name. Default Flickr30k")

parser.add_argument('--seed',type=int, default=1)

parser.add_argument('--tscale',help="Do temperature scailing when measuring calibrity", action="store_true")

parser.add_argument('--noclip',   help="run Group",  action="store_true")
parser.add_argument('--albef',   help="run Group",  action="store_true")
#Multiplier args
parser.add_argument('--vmix',type=float, default=0.1)
parser.add_argument('--lmix',type=float, default=0.1)
parser.add_argument('--vlmix',type=float, default=0.1)
parser.add_argument('--mmmix',type=float, default=0.1)
parser.add_argument('--noise',type=float, default=0.0)
parser.add_argument('--beta1',type=float, default=1.0)
parser.add_argument('--beta2',type=float, default=1.0)
parser.add_argument('--omega',type=float, default=0.1,help="dual multiplier")
parser.add_argument('--zeta',type=float, default=0.1,help="rmse multiplier")
parser.add_argument('--psi',type=float, default=0.0,help="Uniformity multiplier")
parser.add_argument('--mall',type=float, default=0.8,help="ManifoldAll multiplier")
parser.add_argument('--eps',type=float, default=0.005,help="FGSM Epsilon")
parser.add_argument('--neg_mul',type=float, default=0.1,help="NegMix Multiplier")
parser.add_argument('--mode',type=str, default="mix", help="mix or temp")
parser.add_argument('--betavariate',type=float, default=0.2)
parser.add_argument('--geoc',type=float, default=-1.0)
parser.add_argument('--schedule',type=float)

parser.add_argument('--name',type=str, help="Project Name",default="CLIP")
parser.add_argument('--vis',type=str, help="Visualization",default="tsne")
parser.add_argument('--perform_classification', action='store_true')

args = parser.parse_args()

wandb.login(key='1d2a05ae5862d4a852f6e189e5895ec3d50ab0db')
run = wandb.init(project="MM-Mix",allow_val_change=True,name=args.name)
wandb.config.update(args,allow_val_change=True)

import math
import random
import torch
import torchvision
from torchvision import datasets
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from functools import partial
from pathlib import Path

import seaborn as sns

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from CLIP.clip import clip
from CLIP.clip.model import convert_weights, CLIP, mixer_hack
from ALBEF.models.model_retrieval import ALBEF
from ALBEF.models.vit import interpolate_pos_embed

import matplotlib.pyplot as plt

from utils import *
from dataset import COCODataset, FlickerDataset, NFTDataset

import math
from loss import *

import pandas as pd
from tqdm import tqdm

torch.cuda.empty_cache()

torch.Tensor.normalize = lambda x: x/x.norm(dim=-1, keepdim=True)

IS_FIRST = True
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = args.seed

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    np.random.seed(SEED)
    random.seed(SEED)

set_seed(SEED)

#seed setter for Dataloader child process
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def convert_models_to_fp32(model): 
    for p in model.parameters():
        if p.grad is not None :
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

def convert_models_to_fp16(model): 
    for p in model.parameters():
        p.data = p.data.half() 


def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

def do_train(trainloader,clip_model,optimizer,epoch,args,classification_model=None,classification_optimizer=None,logits_scale2=None):
    print("training... 1")
    print(len(trainloader))
    
    temp_num_iters = 0
    
    clip_model.eval() #Clip model needs evalmode for stabilize training..(e.g BN,)
    if classification_model:
        classification_model.train()
    beta = 0.1

    train_loss_acc = 0
    train_loss_classification = 0
    for batch_idx,sample in enumerate(tqdm(trainloader)):
        print("Inside batch")
        images, text_tok, labels    = sample
        # print(f"len sample : {len(sample)}")
        # print(text_tok)
        # print(f"data-> img = {len(images)}, test_tok={len(text_tok)}, label={len(labels)}")
        captions=text_tok
        
        # # print(text_tok)
        # # print(labels)
        # print(images)
        try:
            temp_img = np.array(images)
            assert(temp_img.shape[1] != 0)
            temp_img = list(temp_img.reshape((temp_img.shape[0]*temp_img.shape[1], )))
            images = torch.stack((temp_img))
        except (IndexError, AssertionError) as e:
            print(e)
            print("img-continue")
            continue
        # print(f"temp image shape : {temp_img.shape}")
        # bs, img_lb = temp_img.shape
        # temp_img = list(temp_img.reshape((bs*img_lb, )))
        
        try:
            temp_text = np.array(text_tok)
            assert(temp_text.shape[1] != 0)
            temp_text = list(temp_text.reshape((temp_text.shape[0]*temp_text.shape[1], )))
            text_tok = temp_text
            # print(f'Before tokenization shape: {temp_text}')
        except (IndexError, AssertionError) as e:
            print("text-continue")
            continue
        # # temp_labels = np.array(labels)
        
        # # print(f"np data : img = {temp_img.shape}, txt = {temp_text.shape}, label={temp_labels.shape}")
        
        
        
        if len(captions) != args.bs : break #Drop Last batch

        #Prepare Data Pair. send to GPU.
        images = images.to(device)
        # print(f'Before tokenization: {text_tok}')
        if not args.noclip :
            text_tok = clip.tokenize(text_tok,truncate=True).to(device)

        #Inference
        # print(f'Text tokens = {text_tok}')
        _, _,image_features,text_features = clip_model(images,text_tok)
        
        print(type(image_features), type(text_features))
        print(image_features.shape)
        print(text_features.shape)
        
        image_features = image_features.reshape((args.bs, 5, 512))
        text_features = text_features.reshape((args.bs, 5, 512))

        image_features = torch.mean(image_features, dim=1)
        text_features = torch.mean(text_features, dim=1)
        
        
        #compute Logit
        logits_per_image = image_features@text_features.T
        logits_per_text  = logits_per_image.T

        targets_orig    = torch.eye(len(captions)).to(device) # Identity Matrix

        #Compute Loss. Cross-Entropy with Targets_orig.
        loss    =   torch.zeros([]).to(device)
        loss    +=  clip_loss(
                logits_per_image*(clip_model.logit_scale.exp()/args.divt),
                #logits_per_image/args.divt,
                targets_orig
            )


        I       = targets_orig
        I_R     = torch.flip(I,dims=[0])
        I_D     = 1-I


        def write_original_neg(target,original_neg):
        #This is for NxN Matrix
            cross_I = I+I_R
            cross_I_D = 1 - cross_I
            return target*cross_I + original_neg*cross_I_D

        loss_mix = torch.zeros([]).to(device)

        if epoch > -1 :

            if args.rmse :
                loss_mix += wandb.config.zeta*torch.sqrt(((image_features - text_features)**2).sum())

            #Make Augmented image. reverse order.
            if args.input_cutmix :
                print("Using Cutmix")
                lamb        = torch.rand(1).to(device)
                _, c,image_h, image_w  = images.shape
                cx = random.uniform(0, image_w)
                cy = random.uniform(0, image_h)
                w = image_w * math.sqrt(1 - lamb)
                h = image_h * math.sqrt(1 - lamb)
                x0 = int(round(max(cx - w / 2, 0)))
                x1 = int(round(min(cx + w / 2, image_w)))
                y0 = int(round(max(cy - h / 2, 0)))
                y1 = int(round(min(cy + h / 2, image_h)))

                images_R    = torch.flip(images,dims=[0])
                images_R[:,y0:y1,x0:x1] = images[:,y0:y1,x0:x1]

                logits_per_image, logits_per_text,image_features,text_features = clip_model(images_R,text_tok)
                loss_mix   += calc_mix_loss(logits_per_image,1-lamb,mode=wandb.config.mode)

            #Make Augmented image. reverse order.
            if args.input_img :
                #lamb        = torch.rand(1).to(device)
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                images_R    = torch.flip(images,dims=[0])
                images_A    = lamb*images + (1-lamb)*images_R

                logits_per_image, logits_per_text,image_features,text_features = clip_model(images_A,text_tok)
                logits_per_image *= clip_model.logit_scale.exp()
                loss_mix   += 0.1*calc_mix_loss(logits_per_image,lamb,mode=wandb.config.mode,c=args.geoc,l=1)

            if args.manifold_img :
                #lamb        = torch.rand(1).to(device).half()
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                logits_per_image, logits_per_text,image_features,text_features = clip_model(images,text_tok,mix_image=lamb)
                loss_mix += 0.1*calc_mix_loss(logits_per_image, lamb)

            if args.out_img :
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()

                #pos1 = lamb*image_features + (1-lamb)*torch.flip(image_features,dims=[0])
                pos1 = sph_inter(image_features, torch.flip(image_features,dims=[0]), lamb)
                pos1 = pos1 / pos1.norm(dim=-1,keepdim=True)


                mix_logits = pos1@text_features.T

                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)

                mix_logits = mix_logits*clip_model.logit_scale.exp()/args.divt

                loss_mix += args.vmix*calc_mix_loss(mix_logits,lamb)

            if args.manifold_txt :
                #lamb        = torch.rand(1).to(device).half()
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                #lamb = torch.Tensor([random.betavariate(0.8,0.8)]).to(device).half()
                logits_per_image, logits_per_text,image_features,text_features = clip_model(images,text_tok,mix_text=lamb)
                loss_mix += 0.1*calc_mix_loss(logits_per_image, lamb)

            if args.out_txt :
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()

                #pos1 = lamb*text_features + (1-lamb)*torch.flip(text_features,dims=[0])
                pos1 = sph_inter(text_features, torch.flip(text_features,dims=[0]), lamb)
                pos1 = pos1 / pos1.norm(dim=-1,keepdim=True)


                mix_logits = image_features@pos1.T

                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)

                mix_logits = mix_logits*clip_model.logit_scale.exp()/args.divt

                loss_mix += args.lmix*calc_mix_loss(mix_logits,lamb)

            if args.manifold_all :
                #lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                lamb = torch.Tensor([random.betavariate(args.beta1,args.beta1)]).to(device).half()

                #image_features_mixed    = lamb*image_features + (1-lamb)*torch.flip(image_features,dims=[0])
                #text_features_mixed     = lamb*text_features + (1-lamb)*torch.flip(text_features,dims=[0])
                image_features_mixed = sph_inter(image_features, torch.flip(image_features,dims=[0]), lamb)
                text_features_mixed = sph_inter(text_features, torch.flip(text_features,dims=[0]), lamb)

                image_features_mixed = image_features_mixed.normalize()
                text_features_mixed  = text_features_mixed.normalize()

                mix_logits = image_features_mixed@text_features_mixed.T

                if args.original_neg :
                    mix_logits = write_original_neg(mix_logits,logits_per_image)

                mix_logits = mix_logits*clip_model.logit_scale.exp()/args.divt

                loss_mix += args.vlmix*clip_loss(mix_logits,I)

            if args.dual_mix :
                lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                logits_per_image, logits_per_text,image_features,text_features = clip_model(images,text_tok,dual_mix=lamb)

                logits_per_image2 = logits_per_image*I + (image_features@text_features.T)*I_D
                loss_mix += 0.1*calc_mix_loss(logits_per_image2*clip_model.logit_scale2.exp(), 1)

            if args.neg_mix :
                #lamb = torch.Tensor([random.betavariate(args.betavariate,args.betavariate)]).to(device).half()
                lamb = torch.Tensor([random.betavariate(args.beta2,args.beta2)]).to(device).half()

                #lamb = lamb/2
                targets_orig    = torch.eye(len(captions)).to(device)

                #neg1 = lamb*image_features + (1-lamb)*text_features
                neg1 = sph_inter(image_features, text_features, lamb)
                neg1 = neg1 / neg1.norm(dim=-1,keepdim=True)

                logits_per_image2    = image_features@neg1.T
                logits_per_text2     = text_features@neg1.T

                #logits_per_image2   = image_features@text_features.T
                #logits_per_text2    = text_features@image_features.T

                #logits_per_image2   = torch.cat((logits_per_image,  logits_per_image2*I_D), dim=1)
                #logits_per_text2    = torch.cat((logits_per_text,   logits_per_text2*I_D),  dim=1)
                #targets_orig2       = torch.cat((targets_orig,      targets_orig*0      ),  dim=1)

                #logits_per_image2   = logits_per_image2[:,:-20]
                #logits_per_text2    = logits_per_text2[:,:-20]
                #targets_orig2       = targets_orig2[:,:-20]

                #logits_per_image2 = torch.cat( (logits_per_image,logits_per_image2.flatten().unsqueeze(0).repeat(logits_per_image.shape[0],1)) ,dim=1)
                #logits_per_text2  = torch.cat( (logits_per_text,logits_per_text2.flatten().unsqueeze(0).repeat(logits_per_text.shape[0],1)) ,dim=1)
                #targets_orig2     = torch.cat( (targets_orig, (0*targets_orig).repeat(1,logits_per_text.shape[0])) , dim=1)

                #logits_per_image2   = logits_per_image2[:,:256]
                #logits_per_text2    = logits_per_text2[:,:256]
                #targets_orig2       = targets_orig2[:,:256]


                logits_per_image2   = logits_per_image*I    +   logits_per_image2*I_D
                logits_per_text2    = logits_per_text*I     +   logits_per_text2*I_D
                #print("logits_per_image2_AFTER",logits_per_image2)
                logits_per_image2   = logits_per_image2*clip_model.logit_scale2.exp()/args.divt
                logits_per_text2    = logits_per_text2*clip_model.logit_scale2.exp()/args.divt

                #pdb.set_trace()

                loss_mix           += args.mmmix*clip_loss2(logits_per_image2,I,logits_per_text2,I)
                #loss_mix += torch.nn.functional.cross_entropy(logits_per_image2,torch.arange(image_features.shape[0]).to(device))
                #loss_mix += torch.nn.functional.cross_entropy(logits_per_text2 ,torch.arange(image_features.shape[0]).to(device))

                #loss_mix        += clip_loss2(logits_per_image2,targets_orig) + clip_loss2(logits_per_text2,targets_orig)
            if args.perform_classification:
                # print("In classification")
                # print(f"Shapes : img_features = {image_features.shape}, text_features = {text_features.shape}")
                input_representation = torch.cat([image_features, text_features], dim=1).to(device)
                # print(input_representation.shape, input_representation.dtype)
                input_representation = input_representation.to(torch.float32)
                y_preds = classification_model(input_representation)
                loss_classification = torch.nn.functional.cross_entropy(y_preds, torch.tensor(labels).to(device))
                
                
        temp_num_iters += 1
        if(temp_num_iters > 100):
            print("100 iters done...moving forward")
            break
        print("done this iteration")      
        # break
            #loss += wandb.config.gamma*calc_mix_loss(img_embed_mix_d, txt_embed_mix_d, lamb1*lamb2, (1-lamb1)*(1-lamb2))
        train_loss_acc += ( loss.item() + loss_mix.item())
        train_loss_classification += loss_classification.item()
        wandb.log({"train_loss_iter" : loss.item()})
        wandb.log({"train_MIX_loss_iter" : loss_mix.item()})
        wandb.log({"logit_scale" : (1/clip_model.logit_scale.exp()).item()})
        wandb.log({"logit_scale2" : (1/clip_model.logit_scale2.exp()).item()})
        wandb.log({"train_classifier_loss_iter" : loss_classification.item()})
        
        print(f"Classification loss = {train_loss_classification}")
        print(f"batch loss = {loss_classification.item()}")
        
        if args.method == 'imix':
            loss = loss_mix
        elif args.method == 'unmix':  # not original negative
            loss += loss_mix
        else:
            if not args.schedule :
                loss += (1/(epoch+1))*loss_mix
            elif epoch >= args.epoch*args.schedule :
                loss += loss_mix
        #loss += loss_mix
        #loss = loss_mix
        #if batch_idx%5 == 0 : print(f'Batch idx :{batch_idx},{ loss.item() }, { loss_mix.item() } ')

        optimizer.zero_grad()
        classification_optimizer.zero_grad()
        
        loss.backward(retain_graph = True)
        
        print(f"Before : {loss_classification.dtype}")
        # loss_classification = loss_classification.half()
        # print(f"After : {loss_classification.dtype}")
        loss_classification.backward()
        
        if not args.noclip:
            convert_models_to_fp32(clip_model)
            # convert_models_to_fp32(classification_model)
        
        optimizer.step()
        classification_optimizer.step()
        
        if not args.noclip:
            convert_weights(clip_model)
            # convert_weights(classification_model)
        print(f"done training")




inv_normalize = torchvision.transforms.Normalize(
    mean = [-0.485/0.229*255.0, -0.456/0.224*255.0, -0.406/225*255.0],
    std = [1/0.229,1/0.224,1/0.225]
)
import pdb
def run_calibration(logits_per_image,name,temperature=1.0,bins=10,logWandb=True):
    #Generate Bins
    Buckets = [ [] for i in range(bins)]
    cuts    = [ 1/bins*i for i in range(bins)] + [1]
   
    #Calculate probabillity
    prob = torch.softmax(logits_per_image/temperature,dim=1).cpu()
    #pdb.set_trace()
    maxprob = torch.max(prob,dim=1)

    #Push into Buckets
    for i in range(bins):
        cut_l = cuts[i]
        cut_h = cuts[i+1]
        for mp,y_h,y in zip( maxprob.values.tolist() , maxprob.indices.tolist(),list(range(prob.shape[0])) ) :
            if cut_l < mp and mp <= cut_h :
                #print(f'CUT : {cut_l}, {cut_h}')
                #print(mp,y_h,y)
                Buckets[i].append([mp,y_h,y])

    #Calculate Each Bin's Score
    bucket_accs = []
    bucket_conf = []
    ECE = 0

    #pdb.set_trace()
    for bucket in Buckets :
        tot = len(bucket)
        if tot == 0 :
            acc     = 0
            conf    = 0
        else :
            correct = 0
            mp_sum  = 0
            for mp, y_h, y in bucket :
                if y_h == y : correct +=1
                mp_sum += mp
            acc = correct/tot
            conf = mp_sum/tot
        bucket_accs.append(acc)
        bucket_conf.append(conf)
        ECE += tot/bins*abs(acc-conf)

    #print(f'Bucket Acc for bins={bins},scale={scale} is .. ')
    #print(bucket_accs)
    #print("ECE ERROR")
    #print(ECE)

    """
    if logWandb : 
        plt.clf()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0,1],[0,1],color='black',linestyle='dotted')
        plt.scatter([i/bins for i in range(bins)] , bucket_accs)
        plt.xlabel("confidence")
        plt.ylabel("accuracy")
        # wandb.log({f'{name}_calib': wandb.Image(plt)})
        # wandb.log({F'{name}_ECE' : ECE})
    """
    if logWandb : 
        plt.clf()
        sns.set_style("whitegrid")
        plt.figure(figsize=(6,6))
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0,1],[0,1],color='black',linestyle='--', zorder=1)
        x = [i/bins+0.05 for i in range(bins)]
        for i in range(1,17):
            plt.plot([i*0.05,1],[0,1-0.05*i],color='coral',linestyle='-', zorder=2)
        plt.bar(x , x, width=0.1, color='tomato', alpha=0.35, 
                edgecolor='firebrick', linewidth=2, label='Gap', zorder=3)
        plt.bar(x , bucket_accs, width=0.1, color='mediumturquoise', alpha=1.0, 
                edgecolor='teal', linewidth=2, label='Model Outputs', zorder=4)
        plt.xticks([0.0, 0.5, 1.0], fontsize=14.5)
        plt.yticks([0.0, 0.5, 1.0], fontsize=14.5)
        plt.xlabel(r"$Confidence$", fontsize=17)
        plt.ylabel(r"$Accuracy$", fontsize=17)
        plt.legend(fontsize=14.5)
        #pdb.set_trace()
        plt.savefig(f'{name}_calib.pdf')

        wandb.log({f'{name}_calib': wandb.Image(plt)})
        wandb.log({F'{name}_ECE' : ECE})
    return ECE




def draw_hist(vec,bins=100,name="Sims_Histogram"):
    vec = vec.flatten()
    vec = vec.detach().cpu().numpy()
    plt.clf()
    plt.hist(vec,bins=bins)
    #plt.xlim([-1,1])
    wandb.log({name : wandb.Image(plt)})


g_epoch = 0

def do_valid(validloader,clip_model,optimizer,args,run_calib=True,wandb_prefix="",epoch=0):
    print("Validating...")
    clip_model.eval()
    #convert_models_to_fp32(clip_model) #! evaluation at fp32

    for p in clip_model.parameters():
            p.data = p.data.float() 

    with torch.no_grad():
        valid_loss_acc = 0
        tot_correct = 0
        tot_correct2 = 0
        tot_len = 0
        img_stacks = []
        txt_stacks = []

        for batch_idx, sample in tqdm(enumerate(validloader)):
            images, text_tok, labels = sample
            captions=text_tok
            
            try:
                temp_img = np.array(images)
                assert(temp_img.shape[1] != 0)
                temp_img = list(temp_img.reshape((temp_img.shape[0]*temp_img.shape[1], )))
                images = torch.stack((temp_img))
            except (IndexError, AssertionError) as e:
                print("img-continue")
                continue
        # print(f"temp image shape : {temp_img.shape}")
        # bs, img_lb = temp_img.shape
        # temp_img = list(temp_img.reshape((bs*img_lb, )))
        
            try:
                temp_text = np.array(text_tok)
                assert(temp_text.shape[1] != 0)
                temp_text = list(temp_text.reshape((temp_text.shape[0]*temp_text.shape[1], )))
                text_tok = temp_text
                # print(f'Before tokenization shape: {temp_text}')
            except (IndexError, AssertionError) as e:
                print("txt-continue")
                continue
            
            images      = images.to(device)
            #print(images.min() , images.max())
            images      += args.noise*torch.randn(images.shape).to(device)
            if not args.noclip :
                text_tok    = clip.tokenize(text_tok,truncate=True).to(device)
            logits_per_image, logits_per_text, image_features, text_features = clip_model(images,text_tok)
            img_stacks.append(image_features)
            txt_stacks.append(text_features)

        scale   =  clip_model.logit_scale.exp() 
        scale2  =  clip_model.logit_scale2.exp() 

        image_features = torch.cat(img_stacks,dim=0)
        text_features  = torch.cat(txt_stacks,dim=0)
        logits_per_image = image_features@text_features.T
        #import pdb;pdb.set_trace()
        logits_per_image = logits_per_image * scale
        logits_per_text  = logits_per_image.T

        #pdb.set_trace()

        if run_calib :
            run_calibration(logits_per_image,name="I2T",    temperature=1/scale2)
            run_calibration(logits_per_text,name="T2I",     temperature=1/scale2)

        if args.tscale :
            ts          =     np.arange(0.01,5,0.01)
            I2T_ECES = np.asarray([ run_calibration(logits_per_image,name="I2T",temperature=t,logWandb=False) for t in ts ])
            T2I_ECES = np.asarray([ run_calibration(logits_per_text, name="T2I",temperature=t,logWandb=False) for t in ts ])
            plt.clf()
            plt.plot(ts, I2T_ECES)
            plt.plot(ts, T2I_ECES)
            plt.xlabel("temperature")
            plt.ylabel("ECE Error %")
            plt.ylim([0,20])
            wandb.log({"ECE for scaling" : wandb.Image(plt)})
            
            np.save(f'taudata/{run.name}_epoch{epoch}_{float(np.min(I2T_ECES))}',np.stack([ts,I2T_ECES,T2I_ECES]))
        
        text_sims   =   text_features     @ text_features.T
        imag_sims   =   image_features    @ image_features.T

        text_sims   =   text_sims - text_sims.diag().diagflat()
        imag_sims   =   imag_sims - imag_sims.diag().diagflat()

        all_sims    =   image_features @ text_features.T
        #all_sims_abs = torch.abs(all_sims)
        #pdb.set_trace()

        pair_sims   =   (all_sims).diag()

        #if run_calib :
            #draw_hist(all_sims)

        text_sims = torch.abs(torch.flatten(text_sims))
        imag_sims = torch.abs(torch.flatten(imag_sims))
        pair_sims = torch.abs(torch.flatten(pair_sims))
        neg_sims  = torch.abs(torch.flatten(all_sims.cpu() - all_sims.cpu()*torch.eye(all_sims.shape[0])))

        text_sims_avg = torch.sum(text_sims, dtype=torch.float32)/(text_sims.shape[0]-1000)
        imag_sims_avg = torch.sum(imag_sims, dtype=torch.float32)/(imag_sims.shape[0]-1000)
        pair_sims_avg = torch.sum(pair_sims, dtype=torch.float32)/pair_sims.shape[0]
        neg_sims_avg  = torch.sum(neg_sims , dtype=torch.float32)/neg_sims.shape[0]

        wandb.log({wandb_prefix+"text_sims_avg" : text_sims_avg } )
        wandb.log({wandb_prefix+"imag_sims_avg" : imag_sims_avg } )
        wandb.log({wandb_prefix+"pair_sims_avg" : pair_sims_avg } )
        wandb.log({wandb_prefix+"neg_sims_avg" : neg_sims_avg } )
        wandb.log({wandb_prefix+"sims_ratio" : pair_sims_avg/neg_sims_avg})


        #Accuracy
        I2T = compute_metrics(logits_per_image.cpu().detach().numpy())
        T2I = compute_metrics(logits_per_text.cpu().detach().numpy())
        R1Sum = I2T["R1"] + T2I["R1"]
        print(f'==========================')
        print(f'Image2Text Retrieval : {I2T["R1"]}   {I2T["R5"]}   {I2T["R10"]}')
        print(f'Text2Image Retrieval : {T2I["R1"]}   {T2I["R5"]}   {T2I["R10"]}')
        print(f'==========================')
        I2T = add_key_prefix(I2T,"Valid_I2T_")
        T2I = add_key_prefix(T2I,"Valid_T2I_")



        #if args.dataset == 'coco' :
        #    alignment   = lalign(text_features,image_features )#NOTE!
        #else :
        #    alignment   = rel_lalign(text_features,image_features,relative='diff')

        alignment   = rel_lalign(text_features,image_features,relative='diff')

        #image_unif  = lunif(image_features,t=2)
        #text_unif   = lunif(text_features,t=2)
        mm_unif = lunif(torch.cat([image_features,image_features], dim=0),t=2) #! multimodal uniformity

        wandb.log({wandb_prefix+"Alignment" :alignment.item()} )
        # wandb.log({wandb_prefix+"Imag_Unif" :image_unif.item()} )
        # wandb.log({wandb_prefix+"Text_Unif" :text_unif.item()} )
        wandb.log({wandb_prefix+"Uniformity":mm_unif.item()})


        #print( "Avg Sims")
        ##print( "Text : ",torch.sum(text_sims, dtype=torch.float32)/text_sims.shape[0] )    
        print( "Imag : ",torch.sum(imag_sims, dtype=torch.float32)/imag_sims.shape[0] )    

        #print(compute_metrics(logits_per_image.cpu().detach().numpy()))
        #print(compute_metrics(logits_per_text.cpu().detach().numpy()))
        logits_per_image    = logits_per_image.type(torch.float32)
        #if args.dusoft :
            #logits_per_image = torch.softmax(logits_per_image,dim=0)*torch.softmax(logits_per_image,dim=1)
            #logits_per_image = torch.softmax(logits_per_image*torch.softmax(logits_per_image,dim=0),dim=1)
        #     logits_per_image = logits_per_image*logits_per_image.T
        #     logits_per_text  = logits_per_image.T
        targets_orig        = torch.eye(logits_per_image.shape[0]).to(device)

        final_loss          = clip_loss(logits_per_image,targets_orig)

        wandb.log({wandb_prefix+"valid_loss_iter" : final_loss.item()})


        valid_loss_acc      += final_loss.item()

    if args.vis == 'tsne' :
        tsne_vec(image_features,text_features,"valid")
    else :
        dosnes(image_features,text_features,args.dataset)
    convert_weights(clip_model) #! back to fp16

    return valid_loss_acc ,I2T, T2I, R1Sum

def do_classifier_test(validloader,clip_model,optimizer,args,classifier_model, run_calib=True,wandb_prefix="",epoch=0):
    print("Validating...")
    print(len(validloader))
    clip_model.eval()
    classifier_model.eval()
    #convert_models_to_fp32(clip_model) #! evaluation at fp32

    for p in clip_model.parameters():
            p.data = p.data.float() 

    with torch.no_grad():
        valid_loss_acc = 0
        tot_correct = 0
        tot_correct2 = 0
        tot_len = 0
        img_stacks = []
        txt_stacks = []
        price_movement_preds = []
        y_true = []
        test_loss = 0
        iters = 0
        for batch_idx, sample in tqdm(enumerate(validloader)):
            images, text_tok, labels = sample
            captions=text_tok
            target = torch.tensor(labels).to(device)
            print(f"labels shape : {target.shape}")
            try:
                temp_img = np.array(images)
                assert(temp_img.shape[1] != 0)
                temp_img = list(temp_img.reshape((temp_img.shape[0]*temp_img.shape[1], )))
                images = torch.stack((temp_img))
            except (IndexError, AssertionError) as e:
                print("img-continue")
                continue
        # print(f"temp image shape : {temp_img.shape}")
        # bs, img_lb = temp_img.shape
        # temp_img = list(temp_img.reshape((bs*img_lb, )))
        
            try:
                temp_text = np.array(text_tok)
                assert(temp_text.shape[1] != 0)
                temp_text = list(temp_text.reshape((temp_text.shape[0]*temp_text.shape[1], )))
                text_tok = temp_text
                # print(f'Before tokenization shape: {temp_text}')
            except (IndexError, AssertionError) as e:
                print("txt-continue")
                continue
            
            images      = images.to(device)
            #print(images.min() , images.max())
            images      += args.noise*torch.randn(images.shape).to(device)
            if not args.noclip :
                text_tok    = clip.tokenize(text_tok,truncate=True).to(device)
            logits_per_image, logits_per_text, image_features, text_features = clip_model(images,text_tok)
            image_features = image_features.reshape((args.bs, 5, 512))
            text_features = text_features.reshape((args.bs, 5, 512))

            image_features = torch.mean(image_features, dim=1)
            text_features = torch.mean(text_features, dim=1)
            
            
            
            input_representation = torch.cat([image_features, text_features], dim=1).to(device)
            print(f"Input representation : {input_representation.shape}, {input_representation.dtype}")
            input_representation = input_representation.to(torch.float32)
            y_preds = classifier_model(input_representation)
            price_movement_preds.append(y_preds)
            y_true.append(target)
            # test_loss += torch.nn.functional.cross_entropy(y_preds, target).item()
            # correct += y_preds.eq(target.view_as(pred)).sum().item()
            iters += 1
            if(iters > 100):
                print("------------- 10 iters testing done ----------------")
                break
        # test_loss /= len(validloader.dataset)
        pred_list = torch.cat(price_movement_preds)
        y_true = torch.cat(y_true)
        print(f"before Preds shape : {pred_list.shape}")
        pred_list = torch.argmax(pred_list, axis = 1)
        print(f"after Preds shape : {pred_list.shape}")
        print(f"y_true tensor shape : {y_true.shape}")
        
        pred_list = pred_list.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        print(f"type, len : {type(y_true)}, {y_true.shape}")
        final_acc = (y_true == pred_list).sum() / len(y_true)
        print(f"Accuracy : {final_acc}")
            
            
            
            
def make_fgsm(data,data_grad,epsilon=0.01):
    return data + epsilon*data_grad.sign()


def do_valid_FGSM(validloader,clip_model,optimizer,args):
    print("Validating...FGSM.")
    clip_model.eval()
    with torch.enable_grad():
        valid_loss_acc = 0
        tot_correct = 0
        tot_correct2 = 0
        tot_len = 0

        FGSM_images = []
        text_toks = []
        for batch_idx, sample in enumerate(validloader):
            images, captions = sample
            images      = images.to(device)
            #images      += args.noise*torch.randn(images.shape).to(device)
            text_tok    = clip.tokenize(captions,truncate=True).to(device)
            text_toks.append(text_tok)
            #Requires Grad For FGSM.
            images.requires_grad = True
            #text_tok.requires_grad = True

            logits_per_image, logits_per_text, image_features, text_features = clip_model(images,text_tok,shared=wandb.config.shared,divt=args.divt)
            logits_per_image    = logits_per_image.type(torch.float32)
            targets_orig        = torch.eye(len(captions)).to(device)
            final_loss          = clip_loss(logits_per_image,targets_orig)
            wandb.log({"valid_loss_iter" : final_loss.item()})
            valid_loss_acc      += final_loss.item()
            clip_model.zero_grad()
            final_loss.backward()

            FGSM_images.append(make_fgsm(images,images.grad.data,epsilon=args.eps))
        FGSM_images = torch.cat(FGSM_images)
        text_toks = torch.cat(text_toks)

        print("FGSM_IMAGES SHAPE " ,FGSM_images.shape)
    with torch.no_grad():
        logits_per_image, logits_per_text, image_features, text_features = clip_model(FGSM_images,text_toks,shared=wandb.config.shared,divt=args.divt)

        tsne_vec(image_features,text_features,"FGSM_valid")
        FGSM_I2T = compute_metrics(logits_per_image.cpu().detach().numpy())
        FGSM_T2I = compute_metrics(logits_per_image.T.cpu().detach().numpy())
        print("FGSM I2T R@1 : ",FGSM_I2T["R1"])
        print("FGSM T2I R@1 : ",FGSM_T2I["R1"])
        wandb.log(add_key_prefix(FGSM_I2T,"FGSM_VALID_I2T"))
        wandb.log(add_key_prefix(FGSM_T2I,"FGSM_VALID_T2I"))



def do_SIMAT(clip_model,prep,domain='dev'):
    # get heads !
    DATA_PATH = 'SIMAT/simat_db/images/'
    CLIP_MODEL = 'ViT-B/32'

    model = clip_model
    for p in model.parameters():
            p.data = p.data.float() 

    ds = datasets.ImageFolder(DATA_PATH, transform=prep)

    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=10, shuffle=False)

    img_enc = torch.cat([(model.encode_image2(b.to(device))).cpu().detach() for b, i in tqdm(dl)]).float()

    fnames = [x[0].name for x in datasets.ImageFolder(DATA_PATH, loader=Path)]
    region_ids = [int(x[:-4]) for x in fnames]

    img_enc_mapping = dict(zip(region_ids, img_enc))

    transfos = pd.read_csv('SIMAT/simat_db/transfos.csv', index_col=0)
    words = list(set(transfos.target) | set(transfos.value))
    if not args.noclip :
        tokens = clip.tokenize(words)
        word_encs = torch.cat([(model.encode_text2(b.to(device))).cpu().detach() for b in tqdm(tokens.split(32))])
    else :
        tokens = words
        word_encs = torch.cat([(model.encode_text2(b)).cpu().detach() for b in tqdm(tokens)])

    w2we = dict(zip(words, word_encs))

    emb_key = 'clip'
    #heads = dict(img_head = lambda x:x, txt_head=lambda x:x)
    output = {}
    transfos = pd.read_csv('SIMAT/simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('SIMAT/simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (domain == 'test')]
    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]
    
    #new method
    clip_simat = img_enc_mapping
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float()
    img_embs_stacked = img_embs_stacked.normalize()
    #img_embs_stacked = heads['img_head'](img_embs_stacked).normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    
    word_embs = w2we
    #w2v = {k:heads['txt_head'](v.float()).normalize() for k, v in word_embs.items()}
    w2v = {k:(v.float()).normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = torch.load('SIMAT/simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)
    
    outtt = []
    for lbd in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7]:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5

        
        output[lbd] = float(100*np.average(scores, weights=weights))
        outtt.append(float(100*np.average(scores, weights=weights)))

    print(output)
    return max(outtt)



#Img_encoder : 3x224x224 -> 2048
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder,self).__init__()
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Identity() # Removing last 2048->1000 FC
    
    def forward(self,x):
        return self.img_encoder(x)



class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder,self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def forward(self,x):
        encoded_input = self.tokenizer(x, padding=True ,truncation=True,return_tensors='pt',max_length=200).to(device)
        output = self.model(**encoded_input)
        return output.last_hidden_state[:,0,:] #CLS TOKEN  

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super(ClassificationHead, self).__init__()
        self.hidden_layer = nn.Linear(2 * embedding_dim, 256)
        self.output_layer = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
    
    def forward(self, x):
        x = self.projection(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, img_encoder, txt_encoder, img_emb_dim, txt_emb_dim,joint_emb_dim):
        super().__init__()
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder
        self.img_head    = ProjectionHead(img_emb_dim,projection_dim=joint_emb_dim)
        self.txt_head    = ProjectionHead(txt_emb_dim,projection_dim=joint_emb_dim)
        self.img_emb_dim = img_emb_dim
        self.txt_emb_dim = txt_emb_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def encode_image2(self,img):
        img_feature = self.img_encoder(img)
        img_embed   = self.img_head(img_feature)
        img_embed = img_embed / img_embed.norm(dim=-1,keepdim=True)
        return img_embed

    def encode_text2(self,txt):
        txt_feature = self.txt_encoder(txt)
        txt_embed   = self.txt_head(txt_feature)
        txt_embed = txt_embed / txt_embed.norm(dim=-1,keepdim=True)
        return txt_embed

    def forward(self, img,txt):
        img_embed = self.encode_image2(img)
        txt_embed = self.encode_text2(txt)

        #Calculate loss
        logits_per_image = img_embed@txt_embed.T
        logits_per_text = logits_per_image.T


        return logits_per_image, logits_per_text,img_embed,txt_embed




nft_classification_model = ClassificationHead(embedding_dim=512, n_classes=2)
nft_classification_model.to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device,jit=False,use_shared=wandb.config.shared, prompts_length = 0)
#clip_model, preprocess = clip.load("RN50", device=device,jit=False,use_shared=wandb.config.shared)


if args.noclip :
    img_encoder = ImageEncoder()
    txt_encoder = TextEncoder()

    clip_model = CLIPModel(img_encoder, txt_encoder, 2048, 768,256).cuda()

if args.noclip and args.albef :
    import ruamel.yaml as yaml
    config = yaml.load(open('Retrieval_flickr.yaml', 'r'), Loader=yaml.Loader)
    clip_model = ALBEF(config=config).cuda()
    checkpoint = torch.load("ALBEF_4M.pth", map_location='cpu')
    state_dict = checkpoint['model']

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],clip_model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],clip_model.visual_encoder_m)
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.','')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = clip_model.load_state_dict(state_dict,strict=False)
    clip_model.logit_scale.data = clip_model.temp.data.log()*-1

    print("LOADING \n",msg)

#preprocess= None

if args.weight :
    clip_model.load_state_dict(torch.load(args.weight)['model_state_dict'])

if wandb.config.reinit :
    clip_model.initialize_parameters()

#Copy Mixed Projection Layer
#clip_model.text_projection_mix.data = clip_model.text_projection.data.clone().detach()

#clip_model.logit_scale2.data = clip_model.logit_scale.data.clone().detach()/5

#clip_model.logit_scale.requires_grad = False

"""
print("FREEZE VISUAL ENC")
for n,p in clip_model.named_parameters():
    if 'visual' in n : #n in ['visual']:
        print(n)
        p.requires_grad = False
    else :
        p.requires_grad = True
"""

def custom_collate(list_items):
     x = []
     y = []
     z = []
     for x_, y_, z_ in list_items:
        #  print(f'x_={x_}, y_={y_}')
         x.append(x_)
         y.append(y_)
         z.append(z_)
         
     return x, y, z


if args.dataset == "flickr" :
    trainset    = FlickerDataset('./data/modified_data/flickr30k',transform=preprocess,perc=args.perc).filter_df("train")
    validset    = FlickerDataset('./data/modified_data/flickr30k',transform=preprocess).filter_df("valid")
    testset     = FlickerDataset('./data/modified_data/flickr30k',transform=preprocess).filter_df("test")
elif args.dataset == "coco" :
    trainset    = COCODataset('/data/coco/images/train2017',anon_path='/data/coco/images/annotations/captions_train2017.json',transform=preprocess,perc=args.perc)
    validset    = COCODataset('/data/coco/images/val2017',anon_path='/data/coco/images/annotations/captions_val2017.json',transform=preprocess)
    testset     = COCODataset('/data/coco/images/val2017',anon_path='/data/coco/images/annotations/captions_val2017.json',transform=preprocess)

if args.dataset == "nft":
    trainset = NFTDataset(folder_path = './data/nft_data/half_tweet_data', image_folder_path='./data/nft_data',image_lookback=5, tweet_lookback=5, transform=preprocess).filter_df("train")
    testset = NFTDataset(folder_path = './data/nft_data/half_tweet_data', image_folder_path='./data/nft_data',image_lookback=5, tweet_lookback=5, transform=preprocess).filter_df("test")
    # validset = 



trainloader = DataLoader(trainset, batch_size= wandb.config.bs, collate_fn=custom_collate, shuffle=False)
print("length of train set : " , len(trainset))
# validloader = DataLoader(validset, batch_size= wandb.config.bs, shuffle=False)
testloader  = DataLoader(testset, batch_size= wandb.config.bs, collate_fn=custom_collate, shuffle=False)
testloader2  = DataLoader(testset, batch_size= 128, collate_fn=custom_collate, shuffle=False)


# flickrtest          = FlickerDataset('./data/modified_data/flickr30k',transform=preprocess).filter_df("test")
# cocotest            = COCODataset('/data/coco/images/val2017',anon_path='/data/coco/images/annotations/captions_val2017.json',transform=preprocess)

# flickrtestloader    = DataLoader(flickrtest,    batch_size= wandb.config.bs, shuffle=False)
# cocotestloader      = DataLoader(cocotest,      batch_size= wandb.config.bs, shuffle=False,worker_init_fn=seed_worker)

print("Length : ",len(trainloader))

optimizer = torch.optim.Adam( list(clip_model.parameters()) ,lr=wandb.config.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
scheduler = ExponentialLR(optimizer, gamma=0.9)

nft_classification_optimizer = torch.optim.Adam(nft_classification_model.parameters(), lr=args.classifier_lr, eps=1e-5, weight_decay=0.1)

#Start Training Loop.
Best_R1_sum = -1
for epoch in range(wandb.config.epoch):
    if epoch == 0 :
        optimizer.param_groups[0]['lr'] = wandb.config.startlr #Starting Learning rate 5e-6
    elif epoch == 1 :
        optimizer.param_groups[0]['lr'] = wandb.config.lr


    print("EPOCH ",epoch)
    if args.fgsm :
        do_valid_FGSM(testloader2,clip_model,optimizer,args=args)


    #if args.dataset == "coco":
    #    do_valid(flickrtestloader,clip_model,optimizer,args=args,wandb_prefix="ID-COCO,out-Flickr_")
    #elif args.dataset == "flickr":
    #    do_valid(cocotestloader,clip_model,optimizer,args=args,wandb_prefix="ID-Flickr,out-COCO_")
    #max_simat = do_SIMAT(clip_model,preprocess)
    # wandb.log({"max_simat":max_simat})
    
    do_train(trainloader,clip_model,optimizer,epoch=epoch,args=args, classification_model=nft_classification_model, classification_optimizer=nft_classification_optimizer)
    print("-------------- Training done -------------------")
    # current_best_val,I2T,T2I,R1Sum = do_valid(testloader,clip_model,optimizer,args=args,epoch=epoch)
    do_classifier_test(testloader,clip_model,optimizer, args=args,classifier_model=nft_classification_model, epoch=epoch)

    if R1Sum > Best_R1_sum:
        Best_R1_sum = R1Sum
    wandb.log(I2T)
    wandb.log(T2I)
    wandb.log({"R1_Sum":R1Sum})
    
    scheduler.step()
wandb.log({"Best_R1_Sum" : Best_R1_sum})

torch.save({
        'model_state_dict': clip_model.state_dict(),
    }, f"clip_mod_sph_weight.pt") #just change to your preferred folder/filename



"""
dd2 = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=4)
for sample in dd2 :
    image,caption = sample
    logit2 = clip_model(image,caption)
    plt.imshow(F.softmax(logit2,dim=0).detach().numpy())
    plt.show()
    plt.imshow(F.softmax(logit2,dim=1).detach().numpy())
    plt.show()
    break
"""


# %%
