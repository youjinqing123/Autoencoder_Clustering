import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,                        # download it if you don't have it
)

# plot one example


aim_cluster = np.load('original_60000.npy')
aim_cluster = aim_cluster.reshape(-1, 28, 28)
#label = np.load('original_60000.npy')
label = np.load('aim_cluster_real_60000_new.npy')
#label = np.load('aim_cluster_kmean_60000.npy')

label = label.reshape(-1, 28, 28)


train_data.train_data = torch.tensor(aim_cluster[0:55000,:,:], dtype=torch.uint8)
train_data.train_labels = torch.tensor(label[0:55000,:,:], dtype=torch.uint8)


train_data.train_labels=train_data.train_labels.view(-1,1,28,28)
train_data.train_labels=train_data.train_labels.float()

val_data=torch.tensor(aim_cluster[55000:60000,:,:], dtype=torch.uint8)
val_labels=torch.tensor(aim_cluster[55000:60000,:,:], dtype=torch.uint8)
val_data=val_data.view(-1,1,28,28).float()
val_labels=val_labels.view(-1,1,28,28).float()

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.show()

print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
print(train_data.train_data[2].size())
plt.imshow(train_data.train_labels[2].view(28,28).numpy(), cmap='gray')
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3, padding=1),  # b, 64, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 64, 5, 5
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 32, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1) # b, 32, 2, 2
        )

        self.middle_fu1 = nn.Linear(32 * 2 * 2, 64)

        self.middle_fu2 = nn.Linear(64, 32 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )


    def forward(self, x):
        x=self.encoder(x)
        x = x.view(x.size(0), -1)
        encoded=self.middle_fu1(x)

        x=self.middle_fu2(encoded)
        x = x.view(x.size(0), 32,2,2)
        decoded=self.decoder(x)

        return encoded,decoded
'''

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.middle_fu1 = nn.Linear(8 * 2 * 2, 8)

        self.middle_fu2 = nn.Linear(8, 8 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        encoded = self.middle_fu1(x)

        x = self.middle_fu2(encoded)
        x = x.view(x.size(0), 8, 2, 2)
        decoded = self.decoder(x)

        return encoded, decoded
'''
autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

'''
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
'''

num = 0
save_encode_stack = np.empty((0, 3))
time_train = []
value_train = []
num_train = 0

time_val = []
value_val = []
num_val = 0

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x # batch x, shape (batch, 28*28)
        b_y = y/255.0   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        '''
        imx = b_x[9, :, :, :].view(28, 28).numpy()

        plt.imshow(imx, cmap='gray')
        plt.show()

        imy=b_y[9,:,:,:].view(28,28).numpy()


        plt.imshow(imy, cmap='gray')
        plt.show()

        imx = b_x[19, :, :, :].view(28, 28).numpy()

        plt.imshow(imx, cmap='gray')
        plt.show()

        imy = b_y[19, :, :, :].view(28, 28).numpy()

        plt.imshow(imy, cmap='gray')
        plt.show()

        #b_y=b_y.view(-1,1,28,28)/255
        b_y = b_y.byte()
        '''


        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            time_train.append(num_train)
            num_train += 1
            value_train.append(loss.data.numpy())

            val_x = val_data  # batch x, shape (batch, 28*28)
            val_y = val_labels / 255.0  # batch y, shape (batch, 28*28)
            '''
            imx_val = val_x[9, :, :, :].view(28, 28).numpy()

            plt.imshow(imx_val, cmap='gray')
            plt.show()

            imy_val = val_y[9, :, :, :].view(28, 28).numpy()

            plt.imshow(imy_val, cmap='gray')
            plt.show()

            val_y = val_y.byte()
            '''
            encoded_val, decoded_val = autoencoder(val_x)

            loss_val = loss_func(decoded_val, val_y)

            print('Epoch: ', epoch, '| val loss: %.4f' % loss_val.data.numpy())
            time_val.append(num_val)
            num_val += 1
            value_val.append(loss_val.data.numpy())

            # plotting decoded image (second row)



#torch.save(autoencoder.state_dict(), './mofan_9.pth')
torch.save(autoencoder.state_dict(), './000.pth')

#print(np.shape(save_encode_stack))
#np.save('mofan_0_encode.npy', save_encode_stack)
all_train_data = train_data.train_data.view(-1,1,28,28).type(torch.FloatTensor)/255.
save_encode_stack, _ = autoencoder(all_train_data)
save_encode_stack=save_encode_stack.data.numpy()
print(np.shape(save_encode_stack))
#np.save('mofan_9_encode.npy', save_encode_stack)
np.save('000.npy', save_encode_stack)


plt.figure()
plt.plot(time_train, value_train)
plt.show()

plt.figure()
plt.plot(time_val, value_val)
plt.show()



