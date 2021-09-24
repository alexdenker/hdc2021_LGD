import os 
import pytorch_lightning as pl 

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np 
from torchvision import transforms

from deblurrer.utils import data_util

import torchvision 
import torchvision.transforms as transforms


class BlurredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, font:str='both', num_data_loader_workers:int=8, split:str = 'random'):
        super().__init__()

        assert split in ['random', 'no_split'], 'Currently only a random split (or no split) is implemented'
        assert font in ['both', 'Times', 'Verdana'], 'Only Times, Verdana or both fonts can be used'
        assert blurring_step in np.arange(20), "blurring_step has to an integer between 0 and 9"


        self.batch_size = batch_size    
        self.blurring_step = blurring_step
        self.num_data_loader_workers = num_data_loader_workers
        self.split = split 
        self.font = font


    def prepare_data(self):
        None 

    def setup(self, stage:str = None): 

        if self.font == 'both':
            X_times, Y_times, text_times = data_util.load_data_with_text('Times', self.blurring_step)
            X_verdana, Y_verdana, text_verdana = data_util.load_data_with_text('Verdana', self.blurring_step)

            X = np.concatenate([X_times, X_verdana], axis=0)
            Y = np.concatenate([Y_times, Y_verdana], axis=0)
            text = text_times + text_verdana
        else: 
            X, Y, text = data_util.load_data_with_text(self.font, self.blurring_step)

        # X not blurry, Y blurry
        # X, Y have datatype np.uint16 (Unsigned integer (0 to 65535))
        # Should we do some sort of preprocessing? 

        X = np.expand_dims(X, axis=1).astype(np.float32)
        Y = np.expand_dims(Y, axis=1).astype(np.float32)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        blurred_dataset = BlurredTextDataset(X, Y, text, transform=True)#TensorDataset(X,Y)

        # implement other spliting method?
        if self.split == 'random':
            # use a fixed generator for reproducible results (so random, but fixed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(blurred_dataset, [int(0.8*X.shape[0]), int(0.1*X.shape[0]), int(0.1*X.shape[0])], 
                                                                                    generator=torch.Generator().manual_seed(42))
        elif self.split == 'no_split':
            train_dataset = blurred_dataset
            val_dataset = blurred_dataset
            test_dataset = blurred_dataset

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 

            self.blurred_dataset_train = train_dataset #BlurredDataset(train_dataset, transform=True)
            self.dims = tuple(self.blurred_dataset_train[0][0].shape)

            self.blurred_dataset_validation = val_dataset #BlurredDataset(val_dataset, transform=True)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.blurred_dataset_test = test_dataset #BlurredDataset(test_dataset, transform=True)

            self.dims = tuple(self.blurred_dataset_test[0][0].shape)

    def train_dataloader(self):
            """
            Data loader for the training data.

            Returns
            -------
            DataLoader
                Training data loader.

            """
            return DataLoader(self.blurred_dataset_train, batch_size=self.batch_size,
                            num_workers=self.num_data_loader_workers,
                            shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.blurred_dataset_validation, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.blurred_dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)



class MultipleBlurredDataModule(BlurredDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, font:str='both', num_data_loader_workers:int=8, split:str = 'random'):
        super().__init__(batch_size, blurring_step, font,num_data_loader_workers, split)

    def setup(self, stage:str = None): 

        if self.font == 'both':
            X_times, Y_times, text_times = data_util.load_data_with_text('Times', self.blurring_step)
            X_verdana, Y_verdana, text_verdana = data_util.load_data_with_text('Verdana', self.blurring_step)

            X = np.concatenate([X_times, X_verdana], axis=0)
            Y = np.concatenate([Y_times, Y_verdana], axis=0)
            text = text_times + text_verdana
        else: 
            X, Y, text = data_util.load_data_with_text(self.font, self.blurring_step)

        # X not blurry, Y blurry
        # X, Y have datatype np.uint16 (Unsigned integer (0 to 65535))
        # Should we do some sort of preprocessing? 

        X = np.expand_dims(X, axis=1).astype(np.float32)
        Y = np.expand_dims(Y, axis=1).astype(np.float32)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        blurred_dataset = BlurredTextDataset(X, Y, text, transform=True)#TensorDataset(X,Y)

        # implement other spliting method?
        if self.split == 'random':
            # use a fixed generator for reproducible results (so random, but fixed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(blurred_dataset, [int(0.8*X.shape[0]), int(0.1*X.shape[0]), int(0.1*X.shape[0])], 
                                                                                    generator=torch.Generator().manual_seed(42))
        elif self.split == 'no_split':
            train_dataset = blurred_dataset
            val_dataset = blurred_dataset
            test_dataset = blurred_dataset


        transform_emnist = transforms.Compose(
                [#transforms.Grayscale(), 
                transforms.ToTensor(), 
                transforms.Resize(size=(181, 294)),
                transforms.RandomInvert(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)])

        transform_stl10 = transforms.Compose(
                [transforms.Grayscale(), 
                transforms.ToTensor(), 
                transforms.Resize(size=(181, 294)),
                transforms.RandomInvert(p=0.5)])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 

            self.blurred_dataset_train = train_dataset #BlurredDataset(train_dataset, transform=True)
            self.dims = tuple(self.blurred_dataset_train[0][0].shape)

            self.blurred_dataset_validation = val_dataset #BlurredDataset(val_dataset, transform=True)

            #self.trainset_emnist = torchvision.datasets.EMNIST(root="/localdata/EMNIST", split='balanced', download=False, transform=transform_emnist)
            self.trainset_stl10 =  torchvision.datasets.STL10(root="/localdata/STL10", split='train', download=False, transform=transform_stl10)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.blurred_dataset_test = test_dataset #BlurredDataset(test_dataset, transform=True)

            self.dims = tuple(self.blurred_dataset_test[0][0].shape)

    def train_dataloader(self):

            #trainloader_emnist = torch.utils.data.DataLoader(self.trainset_emnist, batch_size=1,
            #                                        shuffle=True, num_workers=self.num_data_loader_workers)

            #loader_challenge = DataLoader(self.blurred_dataset_train, batch_size=self.batch_size - 2,
            #                num_workers=self.num_data_loader_workers,
            #                shuffle=True, pin_memory=True)
            
           
            #trainloader_stl10 = torch.utils.data.DataLoader(self.trainset_stl10, batch_size=1,
            #                                        shuffle=True, num_workers=self.num_data_loader_workers)


            #loaders = {"EMNIST": trainloader_emnist, "Blurred": loader_challenge, "STL10": trainloader_stl10}

            #return loaders
            return [DataLoader(self.blurred_dataset_train, batch_size=self.batch_size - 2,
                            num_workers=self.num_data_loader_workers,
                            shuffle=True, pin_memory=True), 
                    #DataLoader(self.trainset_emnist, batch_size=1,shuffle=True, num_workers=self.num_data_loader_workers),
                    DataLoader(self.trainset_stl10, batch_size=1,shuffle=True, num_workers=self.num_data_loader_workers)]


class ConcatBlurredDataModule(BlurredDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, font:str='both', num_data_loader_workers:int=8, split:str = 'random'):
        super().__init__(batch_size, blurring_step, font,num_data_loader_workers, split)

    def setup(self, stage:str = None): 

        if self.font == 'both':
            X_times, Y_times, text_times = data_util.load_data_with_text('Times', self.blurring_step)
            X_verdana, Y_verdana, text_verdana = data_util.load_data_with_text('Verdana', self.blurring_step)

            X = np.concatenate([X_times, X_verdana], axis=0)
            Y = np.concatenate([Y_times, Y_verdana], axis=0)
            text = text_times + text_verdana
        else: 
            X, Y, text = data_util.load_data_with_text(self.font, self.blurring_step)

        # X not blurry, Y blurry
        # X, Y have datatype np.uint16 (Unsigned integer (0 to 65535))
        # Should we do some sort of preprocessing? 

        X = np.expand_dims(X, axis=1).astype(np.float32)
        Y = np.expand_dims(Y, axis=1).astype(np.float32)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        blurred_dataset = BlurredTextDataset(X, Y, text, transform=True)#TensorDataset(X,Y)

        # implement other spliting method?
        if self.split == 'random':
            # use a fixed generator for reproducible results (so random, but fixed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(blurred_dataset, [int(0.8*X.shape[0]), int(0.1*X.shape[0]), int(0.1*X.shape[0])], 
                                                                                    generator=torch.Generator().manual_seed(42))
        elif self.split == 'no_split':
            train_dataset = blurred_dataset
            val_dataset = blurred_dataset
            test_dataset = blurred_dataset


        transform_emnist = transforms.Compose(
                [#transforms.Grayscale(), 
                transforms.ToTensor(), 
                transforms.Resize(size=(181, 294)),
                transforms.RandomInvert(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)])

        transform_stl10 = transforms.Compose(
                [transforms.Grayscale(), 
                transforms.ToTensor(), 
                transforms.Resize(size=(181, 294)),
                transforms.RandomInvert(p=0.5)])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 
            trainset_emnist = torchvision.datasets.EMNIST(root="/localdata/EMNIST", split='balanced', download=False, transform=transform_emnist)
            trainset_stl10 =  torchvision.datasets.STL10(root="/localdata/STL10", split='train', download=False, transform=transform_stl10)

            self.blurred_dataset_train = ConcatDataset(train_dataset, trainset_emnist, trainset_stl10)  #BlurredDataset(train_dataset, transform=True)
            self.dims = tuple(train_dataset[0][0].shape)

            self.blurred_dataset_validation = val_dataset #BlurredDataset(val_dataset, transform=True)


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.blurred_dataset_test = test_dataset #BlurredDataset(test_dataset, transform=True)

            self.dims = tuple(self.blurred_dataset_test[0][0].shape)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class BlurredDataset(Dataset):
    """
    Dataset class to include Normalization during training 
    """

    def __init__(self, subset, transform:bool=True):
        self.subset = subset

        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index][0]
        y = self.subset[index][1]
        
        if self.transform: 
            # X, Y have datatype np.uint16 (Unsigned integer (0 to 65535)) - simply divide by 65535
            x = transforms.functional.normalize(x, mean=[0], std=[65535])
            y = transforms.functional.normalize(y, mean=[0], std=[65535])

        return (x,y, self.subset[index][2])
        
    def __len__(self):
        return len(self.subset)

"""
if __name__ == "__main__":
    dataset = BlurredDataModule(font='Times', blurring_step=8)
    dataset.prepare_data()
    dataset.setup()

    for batch in dataset.train_dataloader():
        x, y = batch 
        print(x.shape, y.shape)

"""

class BlurredTextDataset(Dataset):
    def __init__(self, X, Y, true_text, transform:bool=True):
        self.X = X
        self.Y = Y
        self.true_text = true_text

        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        text = self.true_text[index]
        if self.transform: 
            # X, Y have datatype np.uint16 (Unsigned integer (0 to 65535)) - simply divide by 65535
            x = transforms.functional.normalize(x, mean=[0], std=[65535])
            y = transforms.functional.normalize(y, mean=[0], std=[65535])

        return (x,y, text)
        
    def __len__(self):
        return len(self.true_text)



if __name__ == "__main__":
    """
    X_times, Y_times, text_times = data_util.load_data_with_text('Times', 5)
    X_verdana, Y_verdana, text_verdana = data_util.load_data_with_text('Verdana', 5)

    X = np.concatenate([X_times, X_verdana], axis=0)
    Y = np.concatenate([Y_times, Y_verdana], axis=0)

    X = np.expand_dims(X, axis=1).astype(np.float32)
    Y = np.expand_dims(Y, axis=1).astype(np.float32)

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    text = text_times + text_verdana

    print(X.shape, Y.shape, len(text))

    dataset = BlurredTextDataset(X, Y, text)

    x, y, t = dataset[0]
    print(x.shape, y.shape, t)

    import matplotlib.pyplot as plt 
    plt.figure()
    plt.imshow(x[0, :, :], cmap="gray")
    plt.show()
    """
    
    #dataset = MultipleBlurredDataModule(batch_size=12, blurring_step=1)#BlurredDataModule(batch_size=8, blurring_step=step)
    #dataset.prepare_data()
    #dataset.setup()

    #for i in range(2):
    #    for batch in dataset.train_dataloader()['STL10']:
    #        print(batch[0][0,0,50,0])
    #        break
    
    import matplotlib.pyplot as plt 

    dataset = ConcatBlurredDataModule(batch_size=10, blurring_step=4)
    dataset.prepare_data()
    dataset.setup()

    for idx, batch in enumerate(dataset.train_dataloader()):
        print(idx, len(batch))
        print(len(batch[0]), len(batch[1]), len(batch[2]))
        print(batch[0][0].shape)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(batch[1][0][0,0,:,:])
        plt.show()
        
        break