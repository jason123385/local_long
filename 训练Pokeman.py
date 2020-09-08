import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from pokeman import MyPokeman
from resnet18 import Resnet18
from resnet18 import Resnet17
from resnet18 import Resnet16

EPOCH = 20

train_data = MyPokeman(root='./pokeman',resize=224,mode='train')
test_data = MyPokeman(root='./pokeman',resize=224,mode='test')
val_data = MyPokeman(root='./pokeman',resize=224,mode='val')

train_loader = DataLoader(dataset=train_data,batch_size=5,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=5,shuffle=True)
val_loader = DataLoader(dataset=val_data,batch_size=5,shuffle=True)

net = Resnet17()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


best_acc, best_epoch = 0, 0

for epoch in range(EPOCH):
    for step,(x,label) in enumerate(train_loader):
        net.train()
        out = net(x)
        loss = loss_func(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('EPOCH:{},Loss:{}'.format(epoch,loss.item()))

    if epoch % 1 == 0:
        val_acc = evalute(net, val_loader)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc

            torch.save(net.state_dict(), 'Resnet17_5.pkl')

    # correct = 0
    # for x,label in test_loader:
    #     out = net(x)
    #     pred = out.argmax(dim=1)
    #     correct += pred.eq(label).sum().float().item()
    # total_num = len(test_loader.dataset)
    # acc = correct / total_num
    # print('Epoch:{} accruacy:{}'.format(epoch, acc))

# #使用resnet18生成模型
# torch.save(net.state_dict(), 'net1_params.pkl')
# #使用resnet17生成模型
# torch.save(net.state_dict(), 'net2_params.pkl')

# #使用resnet16生成模型
# torch.save(net.state_dict(), 'net3_params.pkl')