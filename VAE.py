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

## 11110COM 526000 Deep Learning HW2:Variational Autoencoder

#======hyperparameters======
batch_size = 50
epochs = 800
lr = 0.0005
#============================

## Don't change the below two functions (compute_PSNR, compute_SSIM)!!
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

def to_var(x):
    if torch.cuda.is_available():
      x = x.cuda()
    return Variable(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#use GPU if available
print(f'Using {device} device')

## Read the data
#total number of data:1476
#shape:(50,50,3) 50*50*3=7500
data = np.load('eye/data.npy')
label = np.load('eye/label.npy')
data = np.reshape(data, (1476,7500))

#create dataset
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

## Your VAE
class VAE(nn.Module):
  def __init__(self):
    super(VAE,self).__init__()
    #encode layers
    self.fc_layer1 = nn.Linear(7500,4000)
    self.fc_layer2 = nn.Linear(4000,2000)
    self.fc_layer3 = nn.Linear(2000,500)
    self.fc_layer4 = nn.Linear(500,100)
    self.fc_layer5_mean = nn.Linear(100,20) #mean
    self.fc_layer5_var = nn.Linear(100,20) #var

    #decode layers
    self.fc_layer6 = nn.Linear(20,100)
    self.fc_layer7 = nn.Linear(100,500)
    self.fc_layer8 = nn.Linear(500,2000)
    self.fc_layer9 = nn.Linear(2000,4000)
    self.fc_layer10 = nn.Linear(4000,7500)

  def encoder(self,x):
    x = F.relu(self.fc_layer1(x)) #F.ReLU(function)
    x = F.relu(self.fc_layer2(x))
    x = F.relu(self.fc_layer3(x))
    x = F.relu(self.fc_layer4(x))
    mean_fc = self.fc_layer5_mean(x) #mean
    var_fc = self.fc_layer5_var(x)  #variance
    return mean_fc, var_fc

  def decoder(self,x):
    x = F.relu(self.fc_layer6(x))
    x = F.relu(self.fc_layer7(x))
    x = F.relu(self.fc_layer8(x))
    x = F.relu(self.fc_layer9(x))
    out = torch.sigmoid(self.fc_layer10(x))
    return out

  def reparameter(self,mu,logvar):
    std = logvar.mul(0.5).exp_()
    esp = to_var(torch.randn(*mu.size()))
    z = mu + std * esp
    return z

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameter(mu, logvar)
    return self.decoder(z), mu, logvar

vae = VAE().to(device)
print(vae)

loss_save = torch.zeros(1, epochs)
loss_save = loss_save.to(device)

## Your training process, loss function and save the torch model in (.pth) format.
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

for epoch in range(epochs):
  for idx, x in enumerate(dataloader):
    x = x.to(device)
    output, mu, logvar = vae(x)

    #calculate reconstruct loss (BCE) and KLD
    BCE = F.binary_cross_entropy(output, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())  
    loss = BCE + KLD

    #update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print loss per epoch    
    if idx%100 == 0:
      print ("Epoch: {}/{}, Reconstruct Loss: {:.4f}".format(epoch+1, epochs, BCE.item()))

    loss_save[0, epoch] += (loss / x.size(0)) #save loss per epoch

loss_save = loss_save.cpu().detach().numpy()

#plot total loss
plt.plot(loss_save.T)
plt.title('VAE total loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss') 
torch.save(vae, 'VAE.pth')

## Your average PSNR, average SSIM on 1476 images and visualization results.
n = 'start'
for batch in dataloader:
  image = batch
  image = image.to(device) 
  output, mu, logvar = vae(image)

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

#check the reconstructed results
original_image = np.reshape(original_image,(1476,50,50,3))
generated_image = np.reshape(generated_image,(1476,50,50,3))
print("check the reconstructed results")
show_num = 5
for i in range(show_num):
    plt.subplot(2,show_num,i+1)
    plt.imshow(original_image[i])
    plt.subplot(2,show_num,i+1+show_num)
    plt.imshow(generated_image[i])

#Add gaussian noise and print
choose_num = [0, 1, 2, 3, 4, 225, 226, 227, 228, 229, 840, 841, 842, 843, 844, 1470, 1471, 1472, 1473, 1474]
choose = data[choose_num,:]
choose = torch.from_numpy(choose).to(torch.float32).to(device)

print_sample = [2, 6, 10, 19] #print 3 227 841 1475 #position from choose_num

for n in range(5) :
    noise = np.random.normal(0,1,(20,20)) #generate noise
    noise = torch.from_numpy(noise).to(torch.float32).to(device)

    #encode
    mu_choose, var_choose = vae.encoder(choose)

    #code
    std_choose = var_choose.mul(0.5).exp_()
    esp_choose = to_var(torch.randn(*mu_choose.size()))
    code_choose = mu_choose + esp_choose * std_choose + noise #add noise
    
    #decode
    output_choose = vae.decoder(code_choose).detach().cpu()

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

sample_label = label[choose_num]
labels = np.zeros((100,1))
for n in range(20) :
    for i in range(5) :
        labels[i + 5*n] = sample_label[n]
print(sample_label[print_sample])

#save
np.save('gen_data.npy',output_choose) #save generated image
np.save('gen_label.npy', labels)

"""reference

Dataset,Dataloader:
https://ithelp.ithome.com.tw/articles/10277163

VAE:https://ithelp.ithome.com.tw/articles/10226549
"""