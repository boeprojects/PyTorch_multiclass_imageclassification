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



```python

```
