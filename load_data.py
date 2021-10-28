from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets, transforms

def mnist(data_path, batch_size, shuffle=True, num_workers=2):
    img_size = 28
    num_class = 10
    train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=data_path, train=True, transform=train_transform, download=True)
    test_data = datasets.MNIST(root=data_path, train=False, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, test_loader, img_size, num_class


def cifar10(data_path, batch_size, shuffle=True, num_workers=2):
    img_size = 32
    num_class = 10
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_loader = DataLoader(datasets.CIFAR10(root=data_path,train=True, download=True, transform=transforms.Compose(
        [transforms.Pad(4,padding_mode='reflect'),transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor(),transforms.Normalize(mean, std)])),
        batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    test_loader = DataLoader(
            datasets.CIFAR10(root=data_path,train=False,download=True,transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])),
        batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, test_loader, img_size, num_class

def svhn(data_path, batch_size, shuffle=True, num_workers=2):
    img_size = 32
    num_class = 10
    mean = [x / 255 for x in [110.1, 109.9, 114.1]]
    std = [x / 255 for x in [50.4, 50.9, 51.2]]
    
    train_data = datasets.SVHN(root=data_path,split='train', download=True, transform=transforms.Compose([transforms.ToTensor()]))
    extra_data = datasets.SVHN(root=data_path,split='extra', download=True, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(ConcatDataset([train_data, extra_data]),
        batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    test_loader = DataLoader(
            datasets.SVHN(root=data_path,split='test',download=True,transform=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])),
        batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, test_loader, img_size, num_class

def fmnist(data_path, batch_size, shuffle=True, num_workers=2):
    img_size = 28
    num_class = 10
    train_loader = DataLoader(datasets.FashionMNIST(root=data_path,train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    test_loader = DataLoader(
            datasets.FashionMNIST(root=data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, test_loader, img_size, num_class

if __name__ =='__main__':
    train_loader, test_loader, img_size, num_class = mnist('/opt/datasets', 128)
    for i, (input_data, labels) in enumerate(train_loader):
        pass
    print(i)