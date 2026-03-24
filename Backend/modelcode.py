# THIS CODE IS WRITTEN IN GOOGLE COLAB AND MODEL TRAINED IN THE GOOGLE COLAB NOTEBOOK AND DATASET IN IN GOOGLE DRIVE

'''
import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from torchvision.datasets import ImageFolder
dataset_path = "/content/drive/MyDrive/Animals"

complete_dataset = ImageFolder(dataset_path,transform=transform)

print("The Classes in Dataset are :- ",complete_dataset.classes,len(complete_dataset))

img,label = complete_dataset[0]
print(img.shape,label)
print(complete_dataset.classes)

train_size = int(0.8*len(complete_dataset))
test_size = len(complete_dataset)-train_size

from torch.utils.data import random_split
train_dataset,test_dataset = random_split(complete_dataset,[train_size,test_size])

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

import torch.nn as nn
class MyNN(nn.Module):
  def __init__(self):
    super(MyNN,self).__init__()
    self.feature_selection = nn.Sequential(
        nn.Conv2d(3,16,3,padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(16,32,3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(32,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2,2)


    )
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(1,3,224,224)
    #dummy = dummy.to(device)
    dummy_in = self.feature_selection(dummy)
    input = torch.numel(dummy_in)

    self.input_feature = input

    self.classifier = nn.Sequential(
        nn.Flatten(),

        nn.Linear(self.input_feature,128),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(128,64),
        nn.BatchNorm1d(64),
        nn.ReLU(),

        nn.Linear(64,32),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.Linear(32,2)
    )

  def forward(self,features):
    features = self.feature_selection(features)
    features = self.classifier(features)
    return features

model = MyNN()
model = model.to(device)

lr = 0.001
epochs = 15

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=lr)
best_loss = float("inf")

for epoch in range(epochs):
  loss_list = []
  f_loss = 0
  for batch_features,batch_labels in train_loader:
    batch_labels = batch_labels.long()
    batch_features,batch_labels = batch_features.to(device),batch_labels.to(device)
    y_pred = model(batch_features)
    loss = loss_fn(y_pred,batch_labels)
    optim.zero_grad()
    loss.backward()
    optim.step()
    loss_list.append(loss)
  f_loss = sum(loss_list)/len(loss_list)
  if(f_loss < best_loss):
    best_loss = f_loss
    torch.save(model.state_dict(),f"/content/drive/MyDrive/Model_save/{epoch+50}.pth")
    print(f"model save at :- /content/drive/MyDrive/Model_save/{epoch+50}.pth")
  print(f"Epochs :- {epoch+1},Loss = {f_loss}")

acc_list = []
with torch.no_grad():
  for batch_features,batch_labels in test_loader:
    acc = 0
    batch_labels = batch_labels.long()
    batch_features,batch_labels = batch_features.to(device),batch_labels.to(device)
    y_pred = model(batch_features)
    y_pred = torch.argmax(y_pred,dim=1)
    acc = (y_pred == batch_labels).float().mean()
    acc_list.append(acc)
  f_acc = sum(acc_list)/len(acc_list)
  print("The accuracy on testing Data is :- ",f_acc.item())
'''