#convert .flac  --> .wav
#cut the wav to 3s

import os
import wave
import librosa
import numpy as np

LibriSpeech_Path = "train-clean-100";
Preprocess_Root_Path = "Preprocess";

sample_rate = 16000;

start_pos_ms = 500;
length_ms = 3000;


def preprocess_handler(input_flac_path,output_wav_path,start_pos_ms,length_ms,sample_rate):
    print(f"{input_flac_path} -> {output_wav_path}");
    data, sr = librosa.load(input_flac_path, sr = sample_rate);
    
    total_length_ms = len(data) / sample_rate * 1000;

    if(total_length_ms >= (start_pos_ms + length_ms)):
        for idx in range (len(data)):
            data[idx] = data[idx] * 65535 / 2;
        data = data.astype(np.int16);
        data = data[int(start_pos_ms * sample_rate / 1000) : int((start_pos_ms + length_ms) * sample_rate / 1000)];
        
        f = wave.open(output_wav_path,'wb');
        f.setnchannels(1);
        f.setsampwidth(2)
        f.setframerate(sample_rate);
        f.writeframes(data.tobytes());
        f.close();

############################################################
speaker_id_dir = os.listdir(LibriSpeech_Path);

for tmp_speaker_dir in speaker_id_dir:

    preprocess_speaker_id_path = os.path.join(Preprocess_Root_Path, tmp_speaker_dir);
    
    if os.path.exists(preprocess_speaker_id_path) == False:
        os.mkdir(preprocess_speaker_id_path);

    speaker_id_dir_path = os.path.join(LibriSpeech_Path, tmp_speaker_dir);
    
    book_id_dir = os.listdir(speaker_id_dir_path);

    for tmp_book_dir in book_id_dir:
        book_id_dir_path = os.path.join(speaker_id_dir_path, tmp_book_dir);

        audio_file_list = os.listdir(book_id_dir_path);

        for tmp_audio_file in audio_file_list:
            last5str = tmp_audio_file[-5:]
            if last5str == ".flac":
                audio_file_full_path = os.path.join(book_id_dir_path, tmp_audio_file);
                preprocess_path = os.path.join(preprocess_speaker_id_path,(tmp_audio_file[:-5] + ".wav"));
                preprocess_handler(audio_file_full_path,preprocess_path,start_pos_ms,length_ms,sample_rate);
                
                