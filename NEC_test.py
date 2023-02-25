import librosa
import scipy.signal as signal
import scipy.fft as fft
import numpy as np

# def Audio_Transform_by_librosa(audio_file):
#     y , sr = librosa.load(audio_file,sr=16000);
#     res = librosa.stft(y,n_fft = 1200,hop_length = 160,win_length = 400,center=True,pad_mode='constant');
#     return res.T;

# def Audio_Transform_by_scipy(audio_file):
#     y , sr = librosa.load(audio_file,sr=16000);
#     f, t, res = signal.stft(y,fs = 16000,window='hann',nperseg=400,noverlap=240,nfft=1200);
#     return res.T;

def Audio_Transform_by_torch(audio_file):
    y , sr = librosa.load(audio_file,sr=16000);
    y = torch.FloatTensor(y);
    res = torch.stft(y,n_fft = 1200, hop_length = 160,win_length = 400,
        window=torch.hann_window(400,False),return_complex = True);
    res = res.numpy();
    return res.T;

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import relu

class Selector(nn.Module):
    def __init__(self) -> None:
        super(Selector,self).__init__();
        self.conv1 = nn.Conv2d(1,64,(1,7),padding = 'same',dilation = (1,1));
        self.conv2 = nn.Conv2d(64,64,(7,1),padding = 'same',dilation = (1,1));
        self.conv3 = nn.Conv2d(64,64,(5,5),padding = 'same',dilation = (1,1));
        self.conv4 = nn.Conv2d(64,64,(5,5),padding = 'same',dilation = (2,1));
        self.conv5 = nn.Conv2d(64,64,(5,5),padding = 'same',dilation = (4,1));
        self.conv6 = nn.Conv2d(64,2,(5,5),padding = 'same',dilation = (8,1));
        self.flatten = nn.Flatten(-2);
        self.linear1 = nn.Linear(1458,600);
        self.linear2 = nn.Linear(600,601);


    def forward(self,x,y):
        x = self.conv1(x);
        # print(f"conv1 shape {x.shape}");
        x = self.conv2(x);
        # print(f"conv2 shape {x.shape}");
        x = self.conv3(x);
        # print(f"conv3 shape {x.shape}");
        x = self.conv4(x);
        # print(f"conv4 shape {x.shape}");
        x = self.conv5(x);
        # print(f"conv5 shape {x.shape}");
        x = self.conv6(x);
        # print(f"conv6 shape {x.shape}");
        x = x.permute(0,2,3,1);
        # print(f"permute shape {x.shape}");

        

        x = self.flatten(x);
        # print(f"flatten shape {x.shape}");
        x = torch.cat([x,y],dim = 2);
        # print(f"cat shape {x.shape}");
        x = relu(self.linear1(x));
        # print(f"linear1 shape {x.shape}");
        x = relu(self.linear2(x));
        # print(f"linear2 shape {x.shape}");
        torch.sigmoid(x);
        # print(f"sigmoid shape {x.shape}");
        return x;

import os
from torch.utils.data import Dataset , DataLoader

Dataset_Train_Path = "NEC_Dataset/train"
Dataset_Valid_Path = "NEC_Dataset/valid"
WAV2MEL_PT_Path = "dvector/wav2mel.pt"
DVECTOR_PT_Path = "dvector/dvector_601.pt"

class NEC_Dataset(Dataset):
    def __init__(self,path):
        speaker_id_dir = os.listdir(path);
        print(speaker_id_dir);
        self.numbers_of_speaker = len(speaker_id_dir);
        print(f"self.numbers_of_speaker = {self.numbers_of_speaker}");

        self.Smixed = [];
        self.Sbg = [];
        self.dvector_repeat = [];

        for tmp_speaker_dir in speaker_id_dir:
            speaker_id_path = os.path.join(path, tmp_speaker_dir);
            
            ref_path = os.path.join(speaker_id_path, "ref");
            ref_dir = os.listdir(ref_path);

            tmp_dvector = torch.randn(1,256);
            tmp_dvector = tmp_dvector.repeat(301,1);

            data_dir_path = os.path.join(speaker_id_path, "data");
            data_dir = os.listdir(data_dir_path);
            for tmp_data_dir in data_dir:
                audio_dir = os.path.join(data_dir_path, tmp_data_dir);
                audio_mixed = os.path.join(audio_dir, "mixed.wav");
                audio_bg = os.path.join(audio_dir, "bg.wav");

                tmp_mixed = torch.from_numpy(Audio_Transform_by_torch(audio_mixed));
                tmp_bg = torch.from_numpy(Audio_Transform_by_torch(audio_bg));
                tmp_mixed = tmp_mixed.real;
                tmp_bg = tmp_bg.real;
                tmp_mixed = tmp_mixed.reshape(301,601);
                


                self.Smixed.append(tmp_mixed);
                self.Sbg.append(tmp_bg);
                self.dvector_repeat.append(tmp_dvector);
            
            print(f"speaker[{tmp_speaker_dir}] load done");

        print(f"dataset len {len(self.Smixed)}");
        

        return;

    def __getitem__(self,index):
        return self.Smixed[index] , self.Sbg[index] , self.dvector_repeat[index];

    def __len__(self):
        return len(self.Smixed);


epoch_total = 50;
learning_rate = 0.001;
log_batch_interval = 5;
save_step_interval = 200;
training_batch_size = 3;

NEC_train_set = NEC_Dataset(Dataset_Train_Path);
NEC_train_dataloader = DataLoader(NEC_train_set,batch_size = training_batch_size,shuffle = True);

NEC_valid_set = NEC_Dataset(Dataset_Valid_Path);
NEC_valid_dataloader = DataLoader(NEC_valid_set,batch_size = 1,shuffle = True);

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
selector = Selector().to(device);
model_loss = nn.MSELoss();
optimizer = torch.optim.SGD(params=selector.parameters(), lr=learning_rate);

step = 0;

# start training
for epoch in range(epoch_total):
    for batch_idx,(Smixed,Sbg,input_dvector) in enumerate(NEC_train_dataloader):
        step += 1;

        Smixed,Sbg,input_dvector = Smixed.cuda(),Sbg.cuda(),input_dvector.cuda();
        optimizer.zero_grad();
        Sshadow = selector(Smixed.unsqueeze(1),input_dvector);

        Srecord = Smixed + Sshadow;

        loss = model_loss(Srecord,Sbg);
        loss.backward();
        
        optimizer.step();

        if batch_idx % log_batch_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, (batch_idx * training_batch_size) + len(Smixed), len(NEC_train_dataloader.dataset),
                            100. * ((batch_idx * training_batch_size) + len(Smixed)) / len(NEC_train_dataloader.dataset),
                            loss.data))
        if step % save_step_interval == 0:
            torch.save(selector,f"selector_step{step}.pth");

