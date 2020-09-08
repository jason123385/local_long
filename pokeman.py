import torch
from torch.utils.data import Dataset
import os
import glob
import random,csv
from torchvision import transforms
from PIL import Image

class MyPokeman(Dataset):
    def __init__(self,root,resize,mode):
        super(MyPokeman,self).__init__()
        self.root = root
        self.resize = resize

        self.name2label = {}
        for item in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root,item)):
                continue

            self.name2label[item] = len(self.name2label.keys())
        # print(self.name2label)
        self.img = self.load_csv('image.csv')

        if mode == 'train':
            self.img = self.img[:int(0.6*len(self.img))]
        elif mode == 'val':
            self.img = self.img[int(0.6*len(self.img)):int(0.8*len(self.img))]
        elif mode == 'test':
            self.img = self.img[int(0.8 * len(self.img)):]

    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root,name,'*.png'))
                images += glob.glob(os.path.join(self.root,name,'*.jpg'))
                images += glob.glob(os.path.join(self.root,name,'*.jpeg'))
            print(len(images),images)
            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    print(name)
                    label = self.name2label[name]
                    print(label)
                    writer.writerow([img,label])

        img = []
        with open(os.path.join(self.root,filename),mode='r') as f:
            reader = csv.reader(f)
            for line in reader:
                image,label = line
                img.append((image,int(label)))

        return img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img, label = self.img[index]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img,label



if __name__ == '__main__':
    db = MyPokeman(root='./pokeman',resize=224,mode='train')
    x,y = next(iter(db))
    print(x.shape,y)

    #特定数据集加载API
    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)