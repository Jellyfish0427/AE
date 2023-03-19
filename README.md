## Autoencoder
### Model architecture  
input -> NN Encoder -> code -> NN Decoder -> output

### Dataset
There are two classes in the dataset, containing female eyes (0) and male eyes (1).  
The total number of data is 1476 and the shape of each image is (50,50,3).  
Example:  
![image](https://user-images.githubusercontent.com/128220508/226117181-b6881e55-f36a-40ae-aa99-f2b31913bad7.png)  
*the dataset file is too big :( 

```js
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
```

### 1. Hyperparameters
```js
batch_size = 50
epochs = 800
lr = 0.0005
```

### 2. Evaluation Metric
- Peak Signal-to-Noise Ratio (PSNR)  
- Structural similarity (SSIM)
```js
def compute_PSNR(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Peak Signal to Noise Ratio (PSNR) function
    # img1 and img2 > [0, 1] 
    
    img1 = torch.as_tensor(img1, dtype=torch.float32)# In tensor format!!
    img2 = torch.as_tensor(img2, dtype=torch.float32)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))

def compute_SSIM(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Structure Similarity (SSIM) function
    # img1 and img2 > [0, 1]
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
```

### 3. Use GPU if available
```js
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
```

### 4. Read data
```js
data = np.load('eye/data.npy')
label = np.load('eye/label.npy')
data = np.reshape(data,(1476,7500))
```

### 5. Create dataset and dataloader
```js
class CreateDataset(Dataset):
  #data loading
  def __init__(self):
    self.data = np.float32(data)
    self.label = np.float32(label)
    self.data_shape = data.shape[0]

  #working for indexing
  def __getitem__(self, index):
    return self.data[index,:]
  
  #return the length of dataset
  def __len__(self):
    return self.data_shape

dataset = CreateDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
```

### 6. AE model
```js
class AE(nn.Module):
  def __init__(self):
    super(AE,self).__init__()
    #encode layers
    self.fc_layer1 = nn.Linear(7500,4000)
    self.fc_layer2 = nn.Linear(4000,2000)
    self.fc_layer3 = nn.Linear(2000,500)
    self.fc_layer4 = nn.Linear(500,100)
    self.fc_layer5 = nn.Linear(100,20)

    #decode layers
    self.fc_layer6 = nn.Linear(20,100)
    self.fc_layer7 = nn.Linear(100,500)
    self.fc_layer8 = nn.Linear(500,2000)
    self.fc_layer9 = nn.Linear(2000,4000)
    self.fc_layer10 = nn.Linear(4000,7500)

  def encoder(self,x):
    x = F.relu(self.fc_layer1(x))
    x = F.relu(self.fc_layer2(x))
    x = F.relu(self.fc_layer3(x))
    x = F.relu(self.fc_layer4(x))
    x = F.relu(self.fc_layer5(x))
    return x

  def decoder(self,x):
    x = F.relu(self.fc_layer6(x))
    x = F.relu(self.fc_layer7(x))
    x = F.relu(self.fc_layer8(x))
    x = F.relu(self.fc_layer9(x))
    out = torch.sigmoid(self.fc_layer10(x))
    return out

  def forward(self, x):
    code = self.encoder(x)
    decode = self.decoder(code)
    return code, decode

ae = AE().to(device)
```

### 7. Training model
```js
loss_save = torch.zeros(1, epochs)
loss_save = loss_save.to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

for epoch in range(epochs):
  for idx, x in enumerate(dataloader):
    #front propergation
    x = x.to(device)
    code, decode = ae(x)
    
    #loss = loss_function(decoded, x)
    BCE = F.binary_cross_entropy(decode, x, size_average=False)
    loss = BCE

    #back propergation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print loss per epoch        
    if idx%100 == 0:
      print ("Epoch: {}/{}, Reconstruct Loss: {:.4f}".format(epoch+1, epochs, loss.item()))
    loss_save[0, epoch] += (loss / x.size(0)) #save loss

loss_save = loss_save.cpu().detach().numpy()
```
reconstruction loss: 211522.46

### 8. Plot total loss
```js
plt.plot(loss_save.T)
plt.title('AE total loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
```
![image](https://user-images.githubusercontent.com/128220508/226115608-598e05b3-98a2-4203-b238-1efc313569e5.png)

### 9. Save model
```js
torch.save(ae, 'AE.pth')
```

### 10. Evaluation
```js
n = 'start'
for batch in dataloader:
  image = batch
  image = image.to(device) #put in GPU if available
  codes, output = ae(image)    #注意隨機分布與shuffle問題

  if n == 'start': #put first batch in original_image and generated_image
    original_image = image
    generated_image = output
    n = 'else'
  else:
    original_image = torch.cat((original_image, image), 0)
    generated_image = torch.cat((generated_image, output), 0)  

original_image = original_image.cpu().numpy()
generated_image = generated_image.cpu().detach().numpy()

#PSNR
psnr = compute_PSNR(original_image, generated_image)
print(psnr)

#SSIM
ssim = compute_SSIM(original_image, generated_image)
print(ssim)
```
SSIM: 0.99  
PSNR: 36.8288

### 11. Check the reconstructed results
```js
original_image = np.reshape(original_image,(1476,50,50,3))
generated_image = np.reshape(generated_image,(1476,50,50,3))

num = 5
for i in range(num):
  plt.subplot(2,num,i+1)
  plt.imshow(original_image[i])
  plt.subplot(2,num,i+1+num)
  plt.imshow(generated_image[i])
```

### 12. Add Gaussian noise and print
Specified the images id as [1-5, 226-230, 841-845, 1471-1475].  
Demonstrate the generated images of [3, 227, 841, 1475].  
```js
#start from 0, specifed numbers -1
choose_num = [0, 1, 2, 3, 4, 225, 226, 227, 228, 229, 840, 841, 842, 843, 844, 1470, 1471, 1472, 1473, 1474]
choose = data[choose_num,:]
choose = torch.from_numpy(choose).to(torch.float32).to(device)

print_sample = [2, 6, 10, 19] #print 3 227 841 1475

for n in range(5) : #generate 5
    noise = np.random.normal(0,1,(20,20))
    noise = torch.from_numpy(noise).to(torch.float32).to(device)

    #encode
    code_choose = ae.encoder(choose) + noise #add noise

    #decoder
    output_choose = ae.decoder(code_choose).detach().cpu()

    #reshape before print
    choose = np.reshape(choose.cpu(),(20,50,50,3))
    output_choose = np.reshape(output_choose,(20,50,50,3))

    num = 4
    plt.figure()
    #print 3, 27, 841, 1475
    for i in range(num):
      p = print_sample[i]
      if n==0:
        plt.subplot(2,num,i+1) #print original image
        plt.imshow(choose[p])
      plt.subplot(2,num,i+1+num) #print generate image 
      plt.imshow(output_choose[p])

    choose = np.reshape(choose,(20,7500))
    choose = choose.to(device)
```
Original images:  
![image](https://user-images.githubusercontent.com/128220508/226116565-67309510-faf1-45c3-9a6b-dc8e0a8ad2c3.png)  
Generated images:  
![image](https://user-images.githubusercontent.com/128220508/226116635-22499711-9502-4292-b0b7-8554c3db8f6f.png)


