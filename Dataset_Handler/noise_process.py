import wave
import numpy as np
from scipy.io import loadmat 

src_sample_rate = 19980

noise_babble_path = "Noise/babble.mat"
noise_factory_path = "Noise/factory2.mat"
noise_vehicle_path = "Noise/volvo.mat"

babble_file_name = "Noise/babble_19980.wav"
factory_file_name = "Noise/factory_19980.wav"
vehicle_file_name = "Noise/vehicle_19980.wav"

babble_file_name_16k = "Noise/babble_16000.wav"
factory_file_name_16k = "Noise/factory_16000.wav"
vehicle_file_name_16k = "Noise/vehicle_16000.wav"

babble_sub_dir = "Noise/Babble"
factory_sub_dir = "Noise/Factory"
vehicle_sub_dir = "Noise/Vehicle"

babble_prefix = "babble"
factory_prefix = "factory"
vehicle_prefix = "vehicle"

####################################################
noise_babble = loadmat(noise_babble_path);
# print(noise_babble.keys());
babble_data = noise_babble['babble'];

babble_data.resize(babble_data.shape[0]);
# print(babble_data.shape);
# print(babble_data);

for idx in range (len(babble_data)):
    babble_data[idx] = babble_data[idx] * 65535 / 2;
babble_data = babble_data.astype(np.int16);
# print(babble_data);

f = wave.open(babble_file_name,'wb');
f.setnchannels(1);
f.setsampwidth(2)
f.setframerate(src_sample_rate);
f.writeframes(babble_data.tobytes());
f.close();

####################################################
noise_factory = loadmat(noise_factory_path);
# print(noise_factory.keys());
factory_data = noise_factory['factory2'];

factory_data.resize(factory_data.shape[0]);
# print(factory_data.shape);
# print(factory_data);

for idx in range (len(factory_data)):
    factory_data[idx] = factory_data[idx] * 65535 / 2;
factory_data = factory_data.astype(np.int16);
# print(factory_data);

f = wave.open(factory_file_name,'wb');
f.setnchannels(1);
f.setsampwidth(2)
f.setframerate(src_sample_rate);
f.writeframes(factory_data.tobytes());
f.close();

####################################################
noise_vehicle = loadmat(noise_vehicle_path);
# print(noise_vehicle.keys());
vehicle_data = noise_vehicle['volvo'];

vehicle_data.resize(vehicle_data.shape[0]);
# print(vehicle_data.shape);
# print(vehicle_data);

for idx in range (len(vehicle_data)):
    vehicle_data[idx] = vehicle_data[idx] * 65535 / 2;
vehicle_data = vehicle_data.astype(np.int16);
# print(vehicle_data);

f = wave.open(vehicle_file_name,'wb');
f.setnchannels(1);
f.setsampwidth(2)
f.setframerate(src_sample_rate);
f.writeframes(vehicle_data.tobytes());
f.close();

########################################################################################################
import librosa

def convert_sample_rate(input_wav_path,output_wav_path,output_sample_rate):
    data, sr = librosa.load(input_wav_path, sr = output_sample_rate);
    
    for idx in range (len(data)):
        data[idx] = data[idx] * 65535 / 2;
    data = data.astype(np.int16);

    f = wave.open(output_wav_path,'wb');
    f.setnchannels(1);
    f.setsampwidth(2)
    f.setframerate(output_sample_rate);
    f.writeframes(data.tobytes());
    f.close();

convert_sample_rate(babble_file_name,babble_file_name_16k,16000);
convert_sample_rate(factory_file_name,factory_file_name_16k,16000);
convert_sample_rate(vehicle_file_name,vehicle_file_name_16k,16000);

########################################################################################################
import os
import wave
import numpy as np

def split_wav(input_wav_path,sub_dir,prefix,split_interval_ms):
    f = wave.open(input_wav_path, 'rb');
    params = f.getparams();
    nchannels, sampwidth, framerate, nframes = params[:4];
    str_data = f.readframes(nframes);
    f.close();

    wave_data = np.frombuffer(str_data, dtype=np.short);
    
    num_of_data_per_wav = int(framerate * split_interval_ms / 1000);
    num_of_wav = int(nframes/num_of_data_per_wav);

    for idx in range(num_of_wav):
        tmpFileName = prefix + f"_split_{idx}.wav";
        tmpFileName = os.path.join(sub_dir, tmpFileName);

        tmpData = wave_data[idx * num_of_data_per_wav : (idx + 1) * num_of_data_per_wav];

        f = wave.open(tmpFileName, 'wb');
        f.setnchannels(nchannels);
        f.setsampwidth(sampwidth);
        f.setframerate(framerate);
        f.writeframes(tmpData.tostring());
        f.close();
        print(f"{tmpFileName} done");



split_wav(babble_file_name_16k,babble_sub_dir,babble_prefix,3000);
split_wav(factory_file_name_16k,factory_sub_dir,factory_prefix,3000);
split_wav(vehicle_file_name_16k,vehicle_sub_dir,vehicle_prefix,3000);