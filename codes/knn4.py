import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.autograd import  Variable
import numpy as np

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
            nn.MaxPool2d(2, stride=1)  # b, 32, 2, 2
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



test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=False,  # download it if you don't have it
)

'''
print(test_data.test_data.size())  # (60000, 28, 28)
print(test_data.test_labels.size())  # (60000)
plt.imshow(test_data.test_data[2].numpy(), cmap='gray')
plt.title('%i' % test_data.test_labels[2])
plt.show()
'''

#first model original-original's test output(10000,64)
#aim_cluster1 = np.load('test_images.npy')
model1 = AutoEncoder()
model1.load_state_dict(torch.load('./000.pth'))
model1.eval()

#aim_cluster1 = aim_cluster1.reshape(aim_cluster1.shape[0], 28, 28)/255
view_data = test_data.test_data.view(10000,1,28,28).type(torch.FloatTensor)/255.

encoded_data, _ = model1(view_data)

save_encode1 = encoded_data.detach().numpy()
print(np.shape(save_encode1))
np.save('009.npy', save_encode1)
