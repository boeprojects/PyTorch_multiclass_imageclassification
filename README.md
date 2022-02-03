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
- Robustheit des Trainings durch Start mit kleiner Learning Rate
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


```python

```
