## Deep Learning in PyTorch multiclass classification computer vision mit Convolutional Network
 - Object Detection am Beispiel des Cifar 10 Datensatzes
 - 5 Training Batches und 1 Test-Batch mit jeweils 10000 Datensätzen
 - Verwendung eines pretrained models als CNN
 - Verwendung von Device Option GPU soweit verfügbar (if-else Bedingung)

### Mounten Google Drive und Laden der Daten

```python
from google.colab import drive

drive.mount("/content/drive")
```
Mounted at /content/drive
```python
%pwd
```
'/content'
```python
%cd drive/MyDrive/PyTorch_Convnet/
```
/content/drive/MyDrive/PyTorch_Convnet
```python
%pwd
```
'/content/drive/MyDrive/PyTorch_Convnet'
```python
%ls
```
cifar-10-batches-py/  cifar-10-python.tar.gz  CIFAR.ipynb
```python
#!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
```python
#!tar -xf cifar-10-python.tar.gz
```
```python
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
```
Configurations
- Einsatz Nvidia Cuda zur Performance Steigerung wenn verfügbar
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```
device(type='cuda')
```python
BATCH_SIZE = 128 #Zum Einstieg/EDA nehmen wir Batch-Size 4, nach erstem Duchlauf auf 128 gesetzt
```
### Feature Scaling - Normalisierung der Daten 
- Skalierung der Eingabevariablen auf kleine Werte zwischen 0 und 1
- Umfang und Störung in den Daten verringern - NN können performant nur kleine Datenwertebereiche verarbeiten
- Nicht skalierte (normalisierte) Eingabevariablen können zu einem langsamen oder instabilen Lernprozess führen, während nicht skalierte Zielvariablen bei Regressionsproblemen dazu führen können, dass Gradienten explodieren und der Lernprozess fehlschlägt.“
### Erzeugung der Pipeline
- Transformation der Bilddaten in einen Tensor
```python
# (x - x.mean()) / x.std() (Normalisierung)

transform = transforms.Compose([
  transforms.ToTensor(), # Umwandlung Bilder in Tensoren
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalisieren Daten (x - 0.5) / 0.5
])
```
### Laden der Daten und Ausgabe eines Samples des Trainsets
```python
from torch.utils.data import DataLoader

trainset = torchvision.datasets.CIFAR10(root="./", train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
trainset[0]

(tensor([[[-0.5373, -0.6627, -0.6078,  ...,  0.2392,  0.1922,  0.1608],
          [-0.8745, -1.0000, -0.8588,  ..., -0.0353, -0.0667, -0.0431],
          [-0.8039, -0.8745, -0.6157,  ..., -0.0745, -0.0588, -0.1451],
          ...,
          [ 0.6314,  0.5765,  0.5529,  ...,  0.2549, -0.5608, -0.5843],
          [ 0.4118,  0.3569,  0.4588,  ...,  0.4431, -0.2392, -0.3490],
          [ 0.3882,  0.3176,  0.4039,  ...,  0.6941,  0.1843, -0.0353]],
 
         [[-0.5137, -0.6392, -0.6235,  ...,  0.0353, -0.0196, -0.0275],
          [-0.8431, -1.0000, -0.9373,  ..., -0.3098, -0.3490, -0.3176],
          [-0.8118, -0.9451, -0.7882,  ..., -0.3412, -0.3412, -0.4275],
          ...,
          [ 0.3333,  0.2000,  0.2627,  ...,  0.0431, -0.7569, -0.7333],
          [ 0.0902, -0.0353,  0.1294,  ...,  0.1608, -0.5137, -0.5843],
          [ 0.1294,  0.0118,  0.1137,  ...,  0.4431, -0.0745, -0.2784]],
 
         [[-0.5059, -0.6471, -0.6627,  ..., -0.1529, -0.2000, -0.1922],
          [-0.8431, -1.0000, -1.0000,  ..., -0.5686, -0.6078, -0.5529],
          [-0.8353, -1.0000, -0.9373,  ..., -0.6078, -0.6078, -0.6706],
          ...,
          [-0.2471, -0.7333, -0.7961,  ..., -0.4510, -0.9451, -0.8431],
          [-0.2471, -0.6706, -0.7647,  ..., -0.2627, -0.7333, -0.7333],
          [-0.0902, -0.2627, -0.3176,  ...,  0.0980, -0.3412, -0.4353]]]), 6)
```
### Split der Validation Daten und Shape
```python
from torch.utils.data import DataLoader

trainset = torchvision.datasets.CIFAR10(root="./", train=True, transform=transform)
trainset, devset = torch.utils.data.random_split(trainset, [45000, 5000])#Split des Validation Dataset!
dataloader = {
    "train": DataLoader(trainset, batch_size=BATCH_SIZE),
    "dev": DataLoader(devset, batch_size=BATCH_SIZE)
}
len(trainset), len(devset)
```
(45000, 5000)
### EDA - Explorative Data Analysis
 - Lookup und Ausgabe Beispielbild (via Denormalisierung)
 - Konzeption eines Torch NN als Vorkonzeption der wesentlichen Parameter, Architektur
 - Erste Konzeption der Shape, des Padding, vor den eigentlichen PyTorch Model
```python
import numpy as np
import matplotlib.pyplot as plt


example_image, example_label = trainset[6]
example_image = example_image.numpy()* 0.5 + 0.5 #Denormalisierung
example_image = np.transpose(example_image, (1, 2, 0)) # Tupel mit Dimensionen (Umgestellt) in (32, 32, 3) [Höhe, Breite, Channel nach Numpy], in PyTroch oft [Channel, Höhe, Breite]
plt.imshow(example_image)
```
![cifar10_02](https://user-images.githubusercontent.com/67191365/152388839-a804fc02-1dd8-48de-a1d9-d531de9d8939.PNG)
```python
example_batch = torch.rand(4, 3, 32, 32)#Batch,Channel,Breite,Höhe
example_batch.shape
```
torch.Size([4, 3, 32, 32])
```python
import torch.nn as nn

conv = nn.Conv2d(3, 2, 3)
conv(example_batch).shape
```
torch.Size([4, 2, 30, 30])
```python
conv1 = nn.Conv2d(3, 6, 5, padding='same') # InChannels, Outchannels, KernelSize(hier immer 5)
conv2 = nn.Conv2d(6, 9, 5, padding='same')
conv3 = nn.Conv2d(9, 6, 5, padding='same')
output = conv1(example_batch)
output = conv2(output)
output = conv3(output)
output.shape
```
torch.Size([4, 6, 32, 32])
### PyTorch Convolutional Model (inkl. Linear Layer)
- Convolutional 2d Layer
- MaxPooling => Robustheit Lernprozess gegenüber Schwankungen, durch Reduktion der Anzahl der Dimensionen der feature Map
- Flatten Layer => Serialisierung der Dimensionalität auf Anzahl der Elemente dieses Tensor auf Inputformat für Linear-Layer (hier 1600)
- Linear Layer
- und ReLU => activation f(x)=max(0,x) (setzt alle neg.Werte auf 0)-aktiviert nur Inputs über einem gewissen Wert (konstanter Gradient der ReLUs resultiert in schnellerem Learning)

Anmerkung: KernelSize 2 ergibt default Stride von 2 in Torch
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.g = nn.ReLU()
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.g(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.g(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.g(x)
        x = self.fc2(x)
        x = self.g(x)
        x = self.fc3(x)
        return x

model = Net().to(device) # Erzeugung des Objekts muss .to_device sein, damit die Configurations oben funktionieren
model

Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=1600, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
  (g): ReLU()
)

model(example_batch.to(device)) # muss ebenfalls .to_device sein (sonst fehler)
model

tensor([[ 1.8028,  0.0142,  0.1897, -0.4372,  0.0413, -2.9302, -0.3038, -3.7052,
          3.8494, -1.2226],
        [ 0.6826, -0.5848,  0.0545, -0.5189, -0.5346, -2.3041, -0.0481, -2.1371,
          2.5746, -0.3297],
        [ 1.2608, -1.6501,  0.9749, -0.4702,  1.6990, -0.7829, -2.2942,  0.0753,
         -0.6522, -1.9353],
        [-0.0977, -0.8777, -0.5549,  0.1869, -0.4560, -2.3032, -0.2049, -1.9178,
          3.6340, -0.6968]], device='cuda:0', grad_fn=<AddmmBackward>)
```
 ### Fitting Cycle
- Split Dev-Set als Validaton Dataset (Trainset bereits vorhanden)
- Update Dataloader mit Dev-Set
- Innerhalb einer supervised Learning Classification aufgabe nutzen wir passend Categorical Crossentropy Loss als Lossfunktion
- Optimizer Adam
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #Robustheit des Trainings durch Start mit kleiner Learning Rate

for epoch in range(10):
    
    for phase in ["train", "dev"]:
        running_loss = 0.0 # loss pro batch in jeweiliger Phase
        if phase == "train":
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # für jeden Mini Batch in der Trainingsphase setzen wir die Gradienten auf Zero zur vollständigen Neuberechnung vor Backpropragation

            outputs = model(inputs)

            loss = criterion(outputs, labels) #Loss Berechnung aus Vergleich von Outputs NN und echten Labels

            if phase == "train": # in Trainingsphase Backpropagation und Optimizer Step
                loss.backward()
                optimizer.step()

            running_loss += loss.item() # aufaddieren des loss zum running-loss

        mean_loss = running_loss / len(dataloader[phase]) # Durchschnitt Loss mit running-loss / Anzahl Batches
        print(f'{epoch} {phase} {mean_loss:.2f}')

0 train 1.67
0 dev 1.43
1 train 1.38
1 dev 1.31
2 train 1.27
2 dev 1.24
3 train 1.18
3 dev 1.19
4 train 1.11
4 dev 1.18
5 train 1.05
5 dev 1.16
6 train 1.00
6 dev 1.14
7 train 0.94
7 dev 1.13
8 train 0.89
8 dev 1.13
9 train 0.85
9 dev 1.14
```
## Ausblick für die Anpassung und Überarbeitung
 - An dieser Stelle fehlt eine Metrik für den Loss, der als Größe hier nicht einschätzbar ist (siehe train, dev Werte in der Epoche) \
   um Aussagen über den Verlauf der Lernkurve treffen zu können.
 - Zeilenweise Summierung des Maximums an definierter Stelle der Werte (.argmax(axis=1) und Berechnung einer Wahrscheinlichkeit in
   torch.sum(softmax(outputs).argmax(axis=1) == labels)/len(labels) => wir errechnen eine accuracy
 - Daraus dann eine mean accuracy, außerdem einen mean Loss!
 - Aktivierungsfunktion für Multiclass classification ist softmax

### Überarbeitung des Fitting Cycle
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
softmax = nn.Softmax()

for epoch in range(10):
    
    for phase in ["train", "dev"]:
        running_loss = 0.0
        running_acc = 0.0
        if phase == "train":
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            acc = torch.sum(softmax(outputs).argmax(axis=1) == labels)/len(labels) #Errechnung der accuracy

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        mean_loss = running_loss / len(dataloader[phase]) # loss pro batch / Anzahl der Batches ergibt mean_loss
        mean_acc = running_acc / len(dataloader[phase]) # accuracy pro batch / Anzahl der Batches ergibt mean_acc
        print(f'{epoch+1} {phase} {mean_loss:.2f} {mean_acc:.2f}') # Ausgabe mean loss und mean accuracy

1 train 0.85 0.70
1 dev 1.19 0.60
2 train 0.81 0.72
2 dev 1.24 0.60
3 train 0.78 0.73
3 dev 1.27 0.59
4 train 0.76 0.73
4 dev 1.29 0.60
5 train 0.74 0.74
5 dev 1.30 0.60
6 train 0.71 0.75
6 dev 1.33 0.60
7 train 0.68 0.76
7 dev 1.37 0.60
8 train 0.66 0.77
8 dev 1.43 0.60
9 train 0.64 0.78
9 dev 1.49 0.59
10 train 0.63 0.78
10 dev 1.51 0.60
```
### Zwischenstand: Validation Werte (dev) gehen hoch - Train Kurve geht runter
- Betrachung der Were genauer: train loss geht runter, train acc geht hoch, dev loss geht hoch, dev acc bewegt sich kaum.
- wir haben eine niedriger werdenden train set error, daher high variance und müssten mehr daten reinbingen
- zudem sind die Error Werte auf hohem Niveau.
### Bessere Strategie ist, auf etablierte, empirisch nachgewiesen wirkungsvollere pretrained models zu setzen vor einer Anpassung der Lernparameter

### Zweiter Durchgang mit ResNet 18 (residual neuronal network)

-	ResNet bietet eine besondere Eignung für image classification
-	Vorteil der Nutzung eines tiefen Neuronalen Netzes (18 Layer in 5 Blöcken) - beschleunigt das Lernen – konzentriert auf den eigentlichen Merkmalsraum
-	Da der Gradient auf vorhergehende Schichten zurückpropagiert wird, kann dieser wiederholte Prozess den Gradienten extrem klein machen oder extrem vergrößern durch     Übergewicht in tiefen NN mit vielen Layern – dies wird durch die residual function neutralisiert und verhindert
-	Die Residuen werden auf Null gesetzt, die Gradientenwerte für einige bestimmte Layer neutralisiert, indem diese übersprungen werden
![ResNet_01](https://user-images.githubusercontent.com/67191365/152990099-a0fbc714-d8d7-4127-aeb1-3e7c42e7c11d.PNG)

### Anpassungen im bisherigen Code dazu:

#### Scaling parameter in format (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
```python
transform = transforms.Compose([
  transforms.ToTensor(), # Umwandlung Bilder in Tensoren
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalisieren Daten (x - 0.5) / 0.5, angepasst für ResNet in 0.485 usw.
])
``` 
#### Für das RestNet wird die learning rate sicherheitshalber angepasst von 0.001 auf 0.0001
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001) # siehe unten
```
#### Der ursprüngliche einfach linar Layer des ResNet Outputs wird durch eine komplexere lineare Architektur ersetzt
```python
model = torchvision.models.resnet18(pretrained=True)
#model.fc = nn.Linear(512, 10)
model.fc = nn.Sequential(# Layer im Resnet ein wenig komplexer gestalten für bessere Ergebnisse.
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
model = model.to(device)
model
``` 
### Architektur/Aufbau des Residual Net
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.g(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.g(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.g(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.g(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = torchvision.models.resnet18(pretrained=True)
#model.fc = nn.Linear(512, 10)
model.fc = nn.Sequential(# Layer im Resnet ein wenig komplexer gestalten für bessere Ergebnisse.
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
model = model.to(device)
model

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=10, bias=True)
)
```
```python
model(example_batch.to(device))

tensor([[-0.3908, -0.0048,  0.0782, -0.1384, -0.3805, -0.3151,  0.6134, -0.2620,
         -0.6883, -1.2653],
        [-1.1035,  0.5565, -0.7254,  0.4781,  0.9075, -1.3814, -0.5302, -1.3947,
         -1.8154,  0.1784],
        [-0.4467, -0.1847,  0.2426,  0.7554,  0.4370,  0.7123,  0.3907, -0.1662,
          0.9159,  0.4060],
        [ 1.3067, -1.1378,  0.0538,  1.9291, -0.1417,  0.0125,  0.7161,  0.7316,
          0.0621,  0.4172]], device='cuda:0', grad_fn=<AddmmBackward0>)
``` 
```python
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) # im ResNet nochmals verkleinerte Learning Rate!
softmax = nn.Softmax()

for epoch in range(50):
    
    for phase in ["train", "dev"]:
        running_loss = 0.0
        running_acc = 0.0
        if phase == "train":
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            acc = torch.sum(softmax(outputs).argmax(axis=1) == labels)/len(labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        mean_loss = running_loss / len(dataloader[phase])# loss pro batch / Anzahl der Batches ergibt mean_loss
        mean_acc = running_acc / len(dataloader[phase])# accuracy pro batch / Anzahl der Batches ergibt mean_acc
        print(f'{epoch+1} {phase} {mean_loss:.2f} {mean_acc:.2f}')

1 train 0.92 0.69
1 dev 0.73 0.75
2 train 0.57 0.81
2 dev 0.71 0.77
3 train 0.44 0.85
3 dev 0.75 0.76
4 train 0.33 0.89
4 dev 0.75 0.78
5 train 0.27 0.91
5 dev 0.91 0.76
6 train 0.21 0.93
6 dev 0.79 0.79
7 train 0.17 0.94
7 dev 0.83 0.78
8 train 0.14 0.95
8 dev 0.83 0.80
9 train 0.12 0.96
9 dev 0.89 0.79
10 train 0.10 0.97
10 dev 0.92 0.79
11 train 0.09 0.97
11 dev 0.88 0.80
12 train 0.08 0.97
12 dev 0.91 0.79
13 train 0.07 0.98
13 dev 0.97 0.79
14 train 0.07 0.98
14 dev 0.97 0.79
15 train 0.06 0.98
15 dev 0.94 0.80
16 train 0.05 0.98
16 dev 0.97 0.80
17 train 0.05 0.98
17 dev 1.08 0.78
18 train 0.05 0.98
18 dev 0.98 0.80
19 train 0.05 0.98
19 dev 1.04 0.80
20 train 0.05 0.98
20 dev 1.09 0.78
21 train 0.05 0.98
21 dev 1.01 0.79
22 train 0.05 0.98
22 dev 0.99 0.80
23 train 0.04 0.99
23 dev 1.13 0.80
24 train 0.03 0.99
24 dev 1.02 0.80
25 train 0.03 0.99
25 dev 1.10 0.79
26 train 0.04 0.99
26 dev 1.07 0.79
27 train 0.04 0.99
27 dev 1.01 0.81
28 train 0.03 0.99
28 dev 1.06 0.80
29 train 0.03 0.99
29 dev 0.98 0.81
30 train 0.03 0.99
30 dev 1.04 0.80
31 train 0.03 0.99
31 dev 0.99 0.81
32 train 0.03 0.99
32 dev 1.06 0.81
33 train 0.03 0.99
33 dev 1.07 0.80
34 train 0.06 0.98
34 dev 0.97 0.81
35 train 0.02 0.99
35 dev 1.12 0.80
36 train 0.03 0.99
36 dev 1.05 0.80
37 train 0.03 0.99
37 dev 1.04 0.81
38 train 0.02 0.99
38 dev 1.13 0.80
39 train 0.02 0.99
39 dev 1.07 0.80
40 train 0.02 0.99
40 dev 1.11 0.80
41 train 0.02 0.99
41 dev 1.15 0.81
42 train 0.02 0.99
42 dev 1.13 0.81
43 train 0.02 0.99
43 dev 1.17 0.81
44 train 0.02 0.99
44 dev 1.08 0.80
45 train 0.02 0.99
45 dev 1.12 0.81
46 train 0.02 0.99
46 dev 1.15 0.80
47 train 0.02 0.99
47 dev 1.14 0.80
48 train 0.02 0.99
48 dev 1.16 0.81
49 train 0.07 0.98
49 dev 1.00 0.81
50 train 0.01 1.00
50 dev 1.09 0.81
```
### Dritter Durchgang mit EfficientNet
- spezialisiert auf CNN und image classification, bietet es bessere Ergebnisse mit wenigen Paramenter und eine geringe Performancelast
- aus diesem Grund setzen wir dieses pretrained model hier ein!
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.g = nn.ReLU() 
        # f(x) = max(0, x)
        # f(1) = max(0, 1) = 1
        # f(-1) = mx(0, -1) = 0
        self.dropout = nn.Dropout(p=0.2)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.g(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.g(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.g(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.g(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

#model = Net().to(device)
model = torchvision.models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
"""model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)"""
model.classifier[0] = nn.Dropout(p=0.4, inplace=True)
model.classifier[1] = nn.Linear(1280, 10)
model = model.to(device)
model

EfficientNet(
  (features): Sequential(
    (0): ConvNormActivation(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SiLU(inplace=True)
    )
    (1): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (2): ConvNormActivation(
            (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0, mode=row)
      )
    )
    (2): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0125, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.025, mode=row)
      )
    )
    (3): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.05, mode=row)
      )
    )
    (4): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0625, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.07500000000000001, mode=row)
      )
      (2): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.08750000000000001, mode=row)
      )
    )
    (5): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1125, mode=row)
      )
      (2): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.125, mode=row)
      )
    )
    (6): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1375, mode=row)
      )
      (1): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.15000000000000002, mode=row)
      )
      (2): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1625, mode=row)
      )
      (3): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.17500000000000002, mode=row)
      )
    )
    (7): Sequential(
      (0): MBConv(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
            (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(inplace=True)
            (scale_activation): Sigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1875, mode=row)
      )
    )
    (8): ConvNormActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SiLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=1)
  (classifier): Sequential(
    (0): Dropout(p=0.4, inplace=True)
    (1): Linear(in_features=1280, out_features=10, bias=True)
  )
)
```
```python
model(example_batch.to(device))

tensor([[-0.1008, -0.8152,  0.2021,  0.0565, -0.0224,  0.5700, -0.4201,  0.3058,
          0.0695,  0.0965],
        [-0.1045, -0.1021, -0.0775, -0.2850, -0.3868, -0.1856,  0.5620, -0.4532,
          0.0157,  0.2833],
        [ 0.3082,  0.3826, -0.0906, -0.2370, -0.4573,  0.0846, -0.0112,  0.6336,
         -0.4652, -0.3983],
        [-0.1353,  0.5469,  0.2281, -0.5138,  0.7700, -0.5427,  0.3245, -0.0136,
          0.1250, -0.3044]], device='cuda:0', grad_fn=<AddmmBackward0>)
```
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Robustheit des Trainings durch Start mit kleiner Learning Rate
softmax = nn.Softmax()

for epoch in range(50):
    
    for phase in ["train", "dev"]:
        running_loss = 0.0
        running_acc = 0.0
        if phase == "train":
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            acc = torch.sum(softmax(outputs).argmax(axis=1) == labels)/len(labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        mean_loss = running_loss / len(dataloader[phase])
        mean_acc = running_acc / len(dataloader[phase])
        print(f'{epoch+1} {phase} {mean_loss:.2f} {mean_acc:.2f}')
        
1 train 1.09 0.62
1 dev 0.68 0.77
2 train 0.62 0.79
2 dev 0.58 0.80
3 train 0.45 0.84
3 dev 0.57 0.82
4 train 0.36 0.87
4 dev 0.61 0.81
5 train 0.29 0.90
5 dev 0.59 0.82
6 train 0.24 0.92
6 dev 0.68 0.81
7 train 0.21 0.93
7 dev 0.67 0.82
8 train 0.18 0.94
8 dev 0.68 0.83
9 train 0.16 0.95
9 dev 0.75 0.82
10 train 0.14 0.95
10 dev 0.66 0.83
11 train 0.13 0.96
11 dev 0.73 0.83
12 train 0.11 0.96
12 dev 0.77 0.83
13 train 0.11 0.96
13 dev 0.67 0.84
14 train 0.10 0.97
14 dev 0.73 0.82
15 train 0.09 0.97
15 dev 0.70 0.84
16 train 0.08 0.97
16 dev 0.81 0.83
17 train 0.09 0.97
17 dev 0.78 0.84
18 train 0.08 0.97
18 dev 0.72 0.84
19 train 0.08 0.97
19 dev 0.76 0.83
20 train 0.07 0.98
20 dev 0.78 0.84
21 train 0.07 0.98
21 dev 0.74 0.84
22 train 0.07 0.98
22 dev 0.72 0.85
23 train 0.07 0.98
23 dev 0.81 0.84
24 train 0.06 0.98
24 dev 0.82 0.84
25 train 0.06 0.98
25 dev 0.76 0.84
26 train 0.06 0.98
26 dev 0.78 0.84
27 train 0.06 0.98
27 dev 0.75 0.84
28 train 0.06 0.98
28 dev 0.78 0.84
29 train 0.05 0.98
29 dev 0.79 0.84
30 train 0.05 0.98
30 dev 0.75 0.83
31 train 0.06 0.98
31 dev 0.84 0.83
32 train 0.05 0.98
32 dev 0.76 0.84
33 train 0.05 0.98
33 dev 0.78 0.84
34 train 0.05 0.98
34 dev 0.81 0.84
35 train 0.05 0.98
35 dev 0.81 0.84
36 train 0.04 0.99
36 dev 0.86 0.84
37 train 0.05 0.99
37 dev 0.76 0.84
38 train 0.04 0.99
38 dev 0.86 0.84
39 train 0.04 0.99
39 dev 0.84 0.84
40 train 0.04 0.99
40 dev 0.80 0.84
41 train 0.04 0.99
41 dev 0.83 0.84
42 train 0.04 0.99
42 dev 0.79 0.83
43 train 0.04 0.99
43 dev 0.85 0.84
44 train 0.04 0.99
44 dev 0.85 0.84
45 train 0.05 0.98
45 dev 0.74 0.84
46 train 0.04 0.99
46 dev 0.85 0.84
47 train 0.03 0.99
47 dev 0.80 0.84
48 train 0.04 0.99
48 dev 0.87 0.84
49 train 0.04 0.99
49 dev 0.91 0.84
50 train 0.04 0.99
50 dev 0.82 0.84

```
