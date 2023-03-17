import torch
from torchvision import transforms
import pandas as pd
import os
from PIL import Image, ImageFile
import albumentations as A
import torchvision
import bisect
import ast
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FlickerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path,transform = None,perc=1.0):
        #Naively, Load all caption data in memory
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.folder_path = folder_path
        self.caption_df = pd.read_csv(os.path.join(self.folder_path,'results.csv')).dropna(axis=0).drop_duplicates(subset="image")
        #Default transform handling
        if transform == None :
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return int(len(self.caption_df)*self.perc)

    def __getitem__(self, idx):

        imgname,caption,type_ = self.caption_df.iloc[idx,:]
        caption = caption
        img = Image.open(os.path.join(self.folder_path,'flickr30k_images',imgname))
        #img = np.asarray(img)
        img = self.transform(img)

        return torch.Tensor(img), caption
    
    def filter_df(self, name):
        self.caption_df = self.caption_df[self.caption_df['type'] == name]
        return self


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,img_folder,anon_path,transform,perc =1.0):
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.ds = torchvision.datasets.CocoCaptions(root=img_folder,annFile=anon_path,transform=transform)
    
    def __len__(self):
        return int(self.perc*len(self.ds))
    
    def __getitem__(self,idx):
        img,caption = self.ds[idx]
        return img, caption[0]

transform_train = \
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


transform_test = \
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self,root,train,transform):
        self.ds = torchvision.datasets.CIFAR100(root=root,train=train,download=True,transform=transform)
        self.class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.prompt = "A photo of "

    def __len__(self):
        return int(len(self.ds))

    def __getitem__(self,idx):
        #import pdb; pdb.set_trace()
        data = self.ds[idx]
        img = data[0]
        #import pdb;pdb.set_trace()
        txt = self.prompt + self.class_names[data[1]]
        return img,txt



class NFTDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, image_folder_path, image_lookback, tweet_lookback, transform=None):
        print("Init dataset")
        self.folder_path = folder_path
        self.transactions_df = pd.read_csv(os.path.join(self.folder_path, 'sorted_price_movement.csv'))
        self.tweets_data = pd.read_csv(os.path.join(self.folder_path, 'sorted_tweet_data_half.csv')).dropna(axis=0)
        self.images_path = os.path.join(image_folder_path, 'images')
        self.image_lookback = image_lookback
        self.tweet_lookback = tweet_lookback
        self.project_mapping = {'CyberKongz':'KONGZ', 
                                'CrypToadz by GREMPLIN': 'TOADZ',
                                'Loot': 'LOOT',
                                'Cool Cats NFT': 'COOL',
                                'World of Women': 'WOW',
                                'BAYC': 'BAYC',
                                'MAYC': 'MAYC',
                                'FLUF World': 'FLUF',
                                'Pudgy Penguins': 'PPG'}
        self.transactions_df = self.transactions_df.loc[self.transactions_df['project'].isin(list(self.project_mapping.keys()))]
        if transform == None :
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return int(len(self.transactions_df))
    
    def __getitem__(self, idx):
        transaction_item = self.transactions_df.iloc[idx, :]
        project = transaction_item['project']
        
        project_tweets = self.tweets_data[self.tweets_data['project'] == project]
        transaction_timestamp = transaction_item['block_timestamp']
        bisect_idx = bisect.bisect_left(project_tweets['datetime'].to_list(), transaction_timestamp)
        
        # tweets_records_list = self.tweets_data.iloc[max(0, bisect_idx - self.tweet_lookback):bisect_idx, :].to_dict('records')
        tweets_text = self.tweets_data.iloc[max(0, bisect_idx - self.tweet_lookback):bisect_idx]['text'].to_list()
        
        
        project_transactions = self.grouped_and_sorted_transactions_df[self.grouped_and_sorted_transactions_df['project'] == project]
        bisect_idx_transactions = bisect.bisect_left(project_transactions['block_timestamp'].to_list(), transaction_timestamp)
        # transactions_list = project_transactions[max(0, bisect_idx_transactions - self.image_lookback):bisect_idx_transactions]['token_ids']
        all_previous_transactions_list = project_transactions[:bisect_idx_transactions]['token_ids']
        
        # project_transactions = self.transactions_df[self.transactions_df['project'] == project].sort_values(by='block_timestamp')
        # bisect_idx_transactions = bisect.bisect_left(project_transactions['block_timestamp'].to_list(), transaction_timestamp)
        # # transactions_list = project_transactions[max(0, bisect_idx_transactions - self.image_lookback):bisect_idx_transactions]['token_ids']
        # all_previous_transactions_list = project_transactions[:bisect_idx_transactions]['token_ids']
        
        images = []
        num_images = 0
        for token_ids_list in all_previous_transactions_list[::-1]:
            if(num_images < self.image_lookback):
                image_ids_list = ast.literal_eval(token_ids_list)
                for image_id in image_ids_list:
                    img_name = self.project_mapping[str(project)] + '_' + str(image_id) + '.png'
                    if os.path.exists(os.path.join(self.images_path, img_name)):
                        img = Image.open(os.path.join(self.images_path, img_name))
                        img = self.transform(img)
                        images.append(torch.Tensor(img))
                        num_images += 1
                        break
            else:
                break
        
        # # print(f"len img: {len(images)}, len tweets: {len(tweets_text)}")
        
        # images = images[:self.image_lookback//2]
        # tweets_text = tweets_text[:self.tweet_lookback//2]
        return images, tweets_text, transaction_item['label']
        
    def filter_df(self, name):
        self.transactions_df = self.transactions_df[self.transactions_df['type'] == name]
        
        projects, counts = np.unique(self.transactions_df['project'].to_numpy(), return_counts=True)
        transactions_dfs = []
        for proj in projects:
        #     # print(proj)
            project_transactions = self.transactions_df[self.transactions_df['project'] == proj]
            project_transactions = project_transactions.sort_values(by='block_timestamp')
            transactions_dfs.append(project_transactions)
        self.grouped_and_sorted_transactions_df = pd.concat(transactions_dfs, ignore_index=True)
        
        return self