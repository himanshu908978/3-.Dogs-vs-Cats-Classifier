import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Model_path = os.path.join(BaseDir,"MODEL","64.pth")
# print(Model_path)
state_dict = torch.load(Model_path,map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

def Inference(inputfile):
  input = Image.open(inputfile)
  input = transform(input)
  input = input.unsqueeze(0)

  with torch.no_grad():
    output = model(input)
    probabilities = torch.softmax(output,dim=1)
    pred_class = torch.argmax(probabilities,dim=1)
    conf = probabilities[0][pred_class]

  return pred_class.tolist()[0], conf.tolist()[0]
