import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from sklearn.model_selection import train_test_split

def data_read(b_n):
    direct = "C:/Users/Administrator/Desktop/data_zip/REDD dataset/house_"+b_n+"/"
    raw_data = pd.read_csv(direct+"channel_1.dat", delimiter= ' ', header= None)
    raw_data = raw_data.to_numpy()[:]
    raw_data2 = pd.read_csv(direct+"channel_2.dat", delimiter= ' ', header= None)
    raw_data2 = raw_data2.to_numpy()[:]
    return raw_data, raw_data2

#Vanilar autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        input_dim = 300
        hidden_dim1 = 200
        hidden_dim2 = 150
        hidden_dim3 = 200
        output_dim = 300

        self.encoder = nn.Sequential(
          nn.Linear(input_dim, hidden_dim1),
          nn.ReLU(True),
          nn.Linear(hidden_dim1, hidden_dim2),
        )
        
        self.decoder = nn.Sequential(
          nn.Linear(hidden_dim2, hidden_dim3),
          nn.ReLU(True),
          nn.Linear(hidden_dim3, output_dim),
        )
                
    def forward(self,x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out
    
class Adap_decoder(nn.Module):
    def __init__(self):
        super(Adap_decoder,self).__init__()
        
        hidden_dim2 = 150
        hidden_dim3 = 200
        output_dim = 300

        self.decoder = nn.Sequential(
          nn.Linear(hidden_dim2, hidden_dim3),
          nn.ReLU(True),
          nn.Linear(hidden_dim3,output_dim),
        )
                
    def forward(self,x):
        out = self.decoder(x)
        return out    

def stan(data):
    data_mean = data.mean(axis = 1).reshape(-1,1)
    data_std = data.std(axis = 1).reshape(-1,1)
    differ = data - data_mean
    differ2 = differ/data_std
    return differ2, data_mean, data_std

def norm(data):
    data_max = data.max(axis = 1).reshape(-1,1)
    data_min = data.min(axis = 1).reshape(-1,1)
    differ = (data-data_min)/(data_max - data_min)
    return differ, data_max, data_min
    
# Device and hyper-parameter setting
batch_size = 128
learning_rate = 0.003
num_epoch = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Making an autoender model, a loss function and an optimizer
model = Autoencoder().to(device)
criterion = nn.L1Loss()
# loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%% Importing dataset

main1, main2 = data_read(str(1))
power_data = main1[:,1]+main2[:,1]

# plt.rcParams['font.size'] = '12'
# plt.figure(dpi = 100)
# plt.plot(power_data, label = 'House 1 profile')
# plt.ylabel('Power [W]')
# plt.xlabel('Sample index')
# plt.legend()

#%% Naming dataset

power_data_tr  = power_data[100001:520001]  # 1400*300
power_data_val = power_data[520001:940001]  # 1400*300
power_data_te  = power_data[940001:1360001] # 1400*300

win_sz  = 300   # window size
num_dt = 1400   # total number of data in dataset 

tr  = power_data_tr.reshape(-1,win_sz)  # train
tr_original  = copy.deepcopy(tr)

val = power_data_val.reshape(-1,win_sz) # validation
val_original = copy.deepcopy(val)

te  = power_data_te.reshape(-1,win_sz)  # test
te_original  = copy.deepcopy(te)

tr  = torch.Tensor(tr)   # change dataform numpy -> tensor
val = torch.Tensor(val)
te  = torch.Tensor(te)

#%% Pre-processing

# Delete noise data  
tr_pp  = tr_original.reshape(-1,win_sz)
val_pp = val_original.reshape(-1,win_sz)
te_pp  = te_original.reshape(-1,win_sz)

check_pp = 0
for dataset_pp in [tr_pp, val_pp, te_pp]:
    
    count, num = 0, 10 # 10 # 0.2385
    dataset_d = np.empty((0,win_sz))
    dataset_large = np.empty((num_dt,win_sz))
    indice_large = np.where(dataset_pp.std(axis=1)>=num)
    indice_small = np.where(dataset_pp.std(axis=1)<num)
    
    for i in range(len(dataset_pp)):
        dataset_concat = np.concatenate((dataset_d,dataset_pp[i].reshape(-1,win_sz)), axis = 0)
        dataset_large[i] = dataset_concat
        dataset_concat = np.empty((0,win_sz))
    dataset_large = np.delete(dataset_large, indice_small, axis = 0)
    
    if check_pp == 0:
        tr_pp_check = copy.deepcopy(dataset_large)
    if check_pp == 1:
        val_pp_check = copy.deepcopy(dataset_large)
    if check_pp == 2:
        te_pp_check = copy.deepcopy(dataset_large)
    check_pp += 1


# Normalization
tr_n, tr_max, tr_min    = norm(tr_pp_check)
tr  = torch.Tensor(tr_n)

val_n, val_max, val_min = norm(val_pp_check)
val = torch.Tensor(val_n)

te_n, te_max, te_min    = norm(te_pp_check)
te  = torch.Tensor(te_n)

#%% Training
tr_dataloader = DataLoader(tr, batch_size=batch_size, shuffle=True)

arr_loss_tr = []
# arr_loss_val = []

for epoch in range(num_epoch):
    for data in tr_dataloader:
        data_tr = data
        data_tr = data_tr.to(device)
        
        # ===================forward=====================
        output = model(data_tr)
        loss = criterion(output, data_tr.data)
        loss_save = loss.detach().item()
        arr_loss_tr.append(loss_save)
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # ===================log========================
    print(f'epoch [{epoch + 1}/{num_epoch}], loss:{loss.item():.4f}')


"""


SAVE THE WEIGHT AND VALUE


"""

tr_or = tr
tr_or = tr_or.detach().numpy().reshape(-1,1)

output_tr  = model(tr)
output_tr = output_tr.detach().numpy().reshape(-1,1)

output_ori = model.forward(tr)
output_ori = output_ori.detach().numpy().reshape(-1,1)

#%%
plt.figure(dpi = 100)
plt.rcParams['font.size'] = '12'
plt.plot(tr_or[3000:5000], label = 'Original data')
plt.plot(output_ori[3000:5000], label = 'Reconstructed data')
plt.ylabel('Power [W]')
plt.xlabel('Sample index')
# plt.legend()


plt.figure(dpi = 100)
plt.rcParams['font.size'] = '12'
plt.xlim([0, num_epoch])
plt.plot(arr_loss_tr, label = 'Training error')
plt.plot(arr_loss_val,label = 'Validation error')
plt.xlabel('Epoch')
plt.ylabel('MAE [kW]')
print('  tr MAE:  {:2f}'.format(arr_loss_tr[-1]))
# print(' val MAE:  {:2f}'.format(val_loss_arr[-1]))
plt.legend()

# val_or = val
# val_output_ori = model.forward(val)
# val_or = val_or.detach().numpy().reshape(-1,1)
# val_output_ori = val_output_ori.detach().numpy().reshape(-1,1)

# te_or = te
# te_output_ori = model.forward(te)
# te_or = te_or.detach().numpy().reshape(-1,1)
# te_output_ori = te_output_ori.detach().numpy().reshape(-1,1)

# loss = loss_func(torch.tensor(te_output_ori),torch.tensor(te_or))
# saving_loss = loss.detach().item()

# print(saving_loss)
#%%

# #training
# plt.figure()
# plt.rcParams['font.size'] = '12'
# plt.figure(dpi = 100)
# plt.plot(tr_or, label = 'Original data')
# plt.plot(output_ori, label = 'Reconstructed data')
# plt.ylabel('Power [W]')
# plt.xlabel('Sample index')
# plt.legend()

# plt.figure(dpi = 100)
# plt.plot(tr_or-output_ori)

# #validation
# plt.figure(dpi = 100)
# plt.plot(val_or, label = 'Original data')
# plt.plot(val_output_ori, label = 'Reconstructed data')
# plt.ylabel('Power [W]')
# plt.xlabel('Sample index')
# plt.legend()

# plt.figure(dpi = 100)
# plt.plot(val_or-val_output_ori)

# #test
# plt.figure(dpi = 100)
# plt.plot(te_or, label = 'Original data')
# plt.plot(te_output_ori, label = 'Reconstructed data')
# plt.ylabel('Power [W]')
# plt.xlabel('Sample index')
# plt.legend()

# plt.figure(dpi = 100)
# plt.plot(te_or-te_output_ori)
# #%%
# lt = model.encoder(tr)
# kmeans = KMeans(n_clusters=4, random_state=0).fit(lt.detach().numpy())
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_
# #%%
# plt.figure()
# plt.plot(centers[0,:])
# plt.plot(centers[1,:])
# plt.plot(centers[2,:])
# plt.plot(centers[3,:])

# center_raw = model.decoder(torch.tensor(centers, dtype = torch.float))
# center_raw = center_raw.detach().numpy()

# plt.figure()
# plt.plot(center_raw[0,:])
# plt.plot(center_raw[1,:])
# plt.plot(center_raw[2,:])
# plt.plot(center_raw[3,:])
# #%%
# dataset_0 = np.empty((0,300))
# dataset_1 = np.empty((0,300))
# dataset_2 = np.empty((0,300))
# dataset_3 = np.empty((0,300))

# tr_or2 = tr_or.reshape(-1,300)
# for i in range(len(tr_or2)):
#     cl_n = labels[i]
#     if cl_n == 0:
#         dataset_0 = np.concatenate((dataset_0,tr_or2[i].reshape(-1,300)), axis = 0)
#     elif cl_n == 1:
#         dataset_1 = np.concatenate((dataset_1,tr_or2[i].reshape(-1,300)), axis = 0)    
#     elif cl_n == 2:
#         dataset_2 = np.concatenate((dataset_2,tr_or2[i].reshape(-1,300)), axis = 0)
#     elif cl_n == 3:
#         dataset_3 = np.concatenate((dataset_3,tr_or2[i].reshape(-1,300)), axis = 0)
# #%%
# plt.figure()
# for i in range(len(dataset_0)):
#     plt.plot(dataset_0[i,:])

# plt.figure()
# for i in range(len(dataset_1)):
#     plt.plot(dataset_1[i,:])

# plt.figure()
# for i in range(len(dataset_2)):
#     plt.plot(dataset_2[i,:])

# plt.figure()
# for i in range(len(dataset_3)):
#     plt.plot(dataset_3[i,:])
# #%%    
# plt.figure()
# for i in range(len(dataset_0)):
#     plt.plot(dataset_0[i,:], alpha = 0.5, color = 'k')
# plt.plot(center_raw[0,:], color = 'r')

# plt.figure()
# for i in range(len(dataset_1)):
#     plt.plot(dataset_1[i,:], alpha = 0.5, color = 'k')
# plt.plot(center_raw[1,:], color = 'r')

# plt.figure()
# for i in range(len(dataset_2)):
#     plt.plot(dataset_2[i,:], alpha = 0.5, color = 'k')
# plt.plot(center_raw[2,:], color = 'r')

# plt.figure()
# for i in range(len(dataset_3)):
#     plt.plot(dataset_3[i,:], alpha = 0.5, color = 'k')
# plt.plot(center_raw[3,:], color = 'r')