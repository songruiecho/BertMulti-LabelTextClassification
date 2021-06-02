from xlnet_utils import *
from myXLNet import *
import torch.optim as optim
import time
import numpy as np
import torch

print(torch.cuda.is_available())
torch.cuda.set_device(0)

torch.manual_seed(1996)
torch.cuda.manual_seed(1996)
np.random.seed(1996)

train_loader, test_loader = load_data()   # function from utils

model = MyXLNet()

print(model)

def eval_net(model):
    model.eval()
    avg_loss = 0.0
    pred_y = []
    labels = []
    for ii, data in enumerate(test_loader):
        X,L = data[0], data[1]
        X = X.long().cuda()
        out = model(X).cpu()
        loss = Mulloss(out, L.float())
        avg_loss+=loss.data.item()
        pred = torch.where(out<0.5, 0, 1).data.numpy().tolist()
        pred_y.extend(pred)
        labels.extend(L.data.numpy().tolist())
    acc = cal_acc(np.array(pred_y), np.array(labels))  # function from utils
    return acc

Mulloss = torch.nn.BCELoss()
best_acc = 0
for epoch in range(1, xlnet_cfg.epoch + 1):
    optimizer = optim.Adam(model.parameters(), lr=xlnet_cfg.lr)
    start = time.time()
    total_loss = 0.0
    for ii, data in enumerate(train_loader):
        model.zero_grad()   # 梯度清零
        model.train()
        X,L = data[0], data[1]
        X = X.long().cuda()
        output = model(X).cpu()
        loss = Mulloss(output, L.float())
        loss.backward()  # 反向传播
        optimizer.step()
        total_loss += loss.data.item()
        end = time.time()
    print('Epoch:{}---------loss:{}-----------time:{}'.format(epoch,total_loss,(end-start)))
    acc = eval_net(model)
    if best_acc < acc:
        best_acc = acc
    print('epoch:{} \t test_acc:{}\tbest_acc:{}'.format(epoch, acc, best_acc))

