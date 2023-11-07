'''
DataLoader for training
'''

import glob
import numpy
import os
import random
import soundfile
import torch
from scipy import signal


class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15],
                         'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = []

        # Use a single loop to find files with the pattern */*/*/*.wav
        for noisecat in os.listdir(musan_path):
            noise_cat_1 = os.path.join(musan_path, noisecat)
            if os.path.isdir(noise_cat_1):
                for noise_cat_2 in os.listdir(noise_cat_1):
                    noise_id = os.path.join(noise_cat_1, noise_cat_2)
                    if os.path.isdir(noise_id):
                        for file_name in os.listdir(noise_id):
                            if file_name.endswith(".wav"):
                                augment_files.append(
                                    os.path.join(noise_id, file_name))

        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                print(file.split('/')[-3])
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = 10*numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10*numpy.log10(numpy.mean(audio ** 2)+1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(
                random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10*numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(
                10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(
            noises, axis=0), axis=0, keepdims=True)
        return 0.1*noise + audio
