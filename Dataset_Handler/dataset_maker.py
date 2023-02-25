import os
import wave
from pydub import AudioSegment
from random import shuffle , randint
from shutil import copyfile

noise_root_path = "Noise";
librispeech_preprocess_root_path = "Preprocess";
dataset_root_path = "NEC_Dataset";

target_speaker_percentage = 0.2

def combine_wav_file(input_path_1,input_path_2,output_path,pos = 0):
    sound1 = AudioSegment.from_wav(input_path_1);
    sound2 = AudioSegment.from_wav(input_path_2);
 
    output = sound1.overlay(sound2,position = pos);  # 把sound2叠加到sound1上面,从第pos秒开始叠加
    output.export(output_path, format="wav");  # 保存文件

# combine_wav_file("19-198-0001.wav","19-198-0009.wav","test.wav");

# file structure of noise
#       noise_root
#       |
#       |
#       |-------noise_type
#       |       |
#       |       |
#       |       |-------noise_file.wav
#       |
#       |-------src.mat
#       |-------src.wav
#       |-------middle.wav

def get_all_noise(path):
    path_list = [];

    noise_type_dir = os.listdir(path);

    for tmp_noise_type_dir in noise_type_dir:
        noise_type_path = os.path.join(path, tmp_noise_type_dir);
        if os.path.isdir(noise_type_path):
            noise_file_list = os.listdir(noise_type_path);
            for tmp_noise_file in noise_file_list:
                path_list.append(os.path.join(noise_type_path, tmp_noise_file));

    return path_list;


# file structure of preprocess LibriSpeech
#       preprocess_root
#       |
#       |
#       |-------speaker_id
#       |       |
#       |       |
#       |       |-------speaker_file.wav

def get_speaker_id_path(path):
    path_list = [];

    speaker_id_dir = os.listdir(path);
    for tmp_speaker_id in speaker_id_dir:
        path_list.append(os.path.join(path, tmp_speaker_id));

    return path_list;


def get_audio_file_from_speaker_id_path(path):
    path_list = [];

    audio_file_list = os.listdir(path);
    for audio_file in audio_file_list:
        path_list.append(os.path.join(path, audio_file));

    return path_list;

def get_shuffle_list(length):
    x = [x for x in range(length)];
    shuffle(x);
    return x;

noise_files = get_all_noise(noise_root_path);
noise_file_total = len(noise_files);
print(f"noise_files total : {noise_file_total}");

speaker_id_list = get_speaker_id_path(librispeech_preprocess_root_path);
print(f"speaker_id total : {len(speaker_id_list)}");

shuffle(speaker_id_list)

number_of_target_speaker = int(target_speaker_percentage * len(speaker_id_list));
print(f"number of target speaker : {number_of_target_speaker}");

target_speaker_idx_list = speaker_id_list[:number_of_target_speaker];
other_speaker_idx_list = speaker_id_list[number_of_target_speaker:];

print(target_speaker_idx_list)

other_speaker_audio_file = [];
for idx in range(len(other_speaker_idx_list)):
    speaker_id_path = other_speaker_idx_list[idx];
    other_speaker_audio_file += get_audio_file_from_speaker_id_path(speaker_id_path);
other_speaker_audio_file_total = len(other_speaker_audio_file);
print(f"other speaker audio files total : {other_speaker_audio_file_total}");



# file structure of NEC dataset
#       NEC_dataset_root
#       |
#       |
#       |-------speaker_id
#               |
#               |
#               |-------ref
#               |       |
#               |       |
#               |       |-------ref0.wav
#               |       |-------ref1.wav
#               |       |-------ref2.wav
#               |
#               |
#               |-------data
#                       |
#                       |
#                       |-------folder_number
#                               |
#                               |
#                               |-------bg.wav
#                               |-------mixed.wav

for idx in range(len(target_speaker_idx_list)):
    speaker_id_path = target_speaker_idx_list[idx];

    audio_file_list = get_audio_file_from_speaker_id_path(speaker_id_path);
    shuffle(audio_file_list);

    speaker_id_dir_name = os.path.split(speaker_id_path)[-1];
    NEC_speaker_id_dir = os.path.join(dataset_root_path, speaker_id_dir_name);
    if os.path.exists(NEC_speaker_id_dir) == False:
        os.mkdir(NEC_speaker_id_dir);

    #REF AUDIO
    NEC_ref_dir = os.path.join(NEC_speaker_id_dir, "ref");
    if os.path.exists(NEC_ref_dir) == False:
        os.mkdir(NEC_ref_dir);

    copyfile(audio_file_list[0], os.path.join(NEC_ref_dir, "ref0.wav"));
    copyfile(audio_file_list[1], os.path.join(NEC_ref_dir, "ref1.wav"));
    copyfile(audio_file_list[2], os.path.join(NEC_ref_dir, "ref2.wav"));


    #DATA AUDIO
    NEC_data_dir = os.path.join(NEC_speaker_id_dir, "data");
    if os.path.exists(NEC_data_dir) == False:
        os.mkdir(NEC_data_dir);

    for file_idx in range(3,len(audio_file_list)):
        folder_dir = os.path.join(NEC_data_dir, f"{file_idx - 3}");
        if os.path.exists(folder_dir) == False:
            os.mkdir(folder_dir);

        noise_file_idx = randint(0,noise_file_total - 1);
        other_file_idx = randint(0,other_speaker_audio_file_total - 1);

        noise_file = noise_files[noise_file_idx];
        other_file = other_speaker_audio_file[other_file_idx];
        target_file = audio_file_list[file_idx];

        bg_file = os.path.join(folder_dir, "bg.wav");
        mixed_file = os.path.join(folder_dir, "mixed.wav");

        combine_wav_file(noise_file,other_file,bg_file);
        combine_wav_file(target_file,bg_file,mixed_file);