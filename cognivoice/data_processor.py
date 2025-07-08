import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
import torchaudio.transforms as at
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor, BertTokenizer


sex_mapping = {'F': 0, 'M': 1}
dx_mapping = {'NC': 0, 'MCI': 1}
lang_mapping = {'en': 0, 'cn': 1}
disvoice_feature_names = ['static-Articulation', 'static-Phonation',
       'static-RepLearning', 'static-Prosody', 'static-Phonological',
       'dynamic-Articulation', 'dynamic-Phonation', 'dynamic-RepLearning',
       'dynamic-Prosody', 'dynamic-Phonological']

def load_wave(wave_path, sample_rate:int=16000):
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


class TAUKADIALDataset(Dataset):
    def __init__(self, args, subset=None):
        super().__init__()

        self.args = args
        self.data = pd.read_csv('/content/drive/MyDrive/TAUKADIAL-24/train/groundtruth.csv')  #/data/datasets/TAUKADIAL-24/train/groundtruth.csv
        print(f"Initial groundtruth.csv data shape: {self.data.shape}")
        print(f"Initial groundtruth 'dx' distribution:\n{self.data['dx'].value_counts()}")

        disvoice = pd.read_parquet('/content/drive/MyDrive/TAUKADIAL-24_feat/train/feats_train.parquet') #/data/datasets/TAUKADIAL-24/feature/feats_train.parquet
        print(f"Disvoice features loaded shape: {disvoice.shape}")
        
        for i in disvoice.columns:
            if i != 'filename':
                disvoice[i+'_mean'] = disvoice[i].apply(np.mean).fillna(0)

        self.data = pd.merge(self.data, disvoice, left_on='tkdname', right_on='filename')
        print(f"Shape after merging with disvoice: {self.data.shape}")
        print(f"'dx' distribution after merging with disvoice:\n{self.data['dx'].value_counts()}")

        if subset is not None:
            self.data = self.data.iloc[subset]
            print(f"Shape after applying subset: {self.data.shape}")
            print(f"'dx' distribution after applying subset:\n{self.data['dx'].value_counts()}")

        self.data['audio'] = [load_wave(f'/content/drive/MyDrive/TAUKADIAL-24/train/{i}', sample_rate=self.args.sample_rate).flatten() for i in self.data.tkdname] #/data/datasets/TAUKADIAL-24/train/{i}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.args.method)

        # Add check for empty audio
        empty_audio_count = sum(1 for a in self.data['audio'] if a.numel() == 0)
        print(f"Number of empty audio waveforms loaded: {empty_audio_count}")
        if empty_audio_count > 0:
            # You might want to remove these rows or handle them specifically
            # For now, just identify them
            empty_audio_tkdnames = self.data.loc[[a.numel() == 0 for a in self.data['audio']], 'tkdname'].tolist()
            print(f"TKD names with empty audio: {empty_audio_tkdnames}")
        empty_input_features_count = sum(1 for f in self.data['input_features'] if f.numel() == 0)
        print(f"Number of empty input_features after extraction: {empty_input_features_count}")

        if 'whisper' in self.args.method:
            self.data['input_features'] = [
                self.feature_extractor(i, sampling_rate=self.args.sample_rate, return_tensors="pt")['input_features'][0]
                for i in self.data['audio']
            ]
        
        elif 'wav2vec2' in self.args.method:
            self.data['input_features'] = [
                self.feature_extractor(
                    i,
                    max_length=self.args.max_length,
                    sampling_rate=self.args.sampling_rate,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")['input_values'][0]
                for i in self.data['audio']
            ]

        self.en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # Transcribed text=
        text = pd.read_parquet('/content/drive/MyDrive/TAUKADIAL-24/translation_train.parquet') #/data/datasets/TAUKADIAL-24/transcription/translation_train.parquet
        print(f"Text data loaded shape: {text.shape}")
        self.data = pd.merge(self.data, text, left_on='tkdname', right_on='file_name')
        print(f"Shape after merging with text: {self.data.shape}")
        print(f"'dx' distribution after merging with text:\n{self.data['dx'].value_counts()}")
        
        # Check for empty text features
        empty_text_input_ids_count = sum(1 for iid in self.data['text_input_ids'] if not iid) # check if list/tensor is empty
        print(f"Number of empty text input_ids after tokenization: {empty_text_input_ids_count}")

        text_feat = [
            self.en_tokenizer(j, padding='max_length', max_length=256, truncation=True) if i == 'en' else
            self.cn_tokenizer(j, padding='max_length', max_length=256, truncation=True)
            for i, j in zip(self.data.language, self.data.transcribed_text)
        ]
        self.data['text_input_ids'] = [i['input_ids'] for i in text_feat]
        self.data['text_attention_mask'] = [i['attention_mask'] for i in text_feat]

        # LLAMA-2 explainations
        #self.data['pid'] = self.data.tkdname.apply(lambda x: x.split('-')[1])
        #self.data['llama2'] = self.data.pid.apply(lambda x: open(f'/data/datasets/TAUKADIAL-24/llama2/llama2_train/{x}.txt').read())

        #llama2_feat = self.en_tokenizer(self.data['llama2'].tolist(), padding='max_length', max_length=256, truncation=True)
        #self.data['llama2_input_ids'] = llama2_feat['input_ids']
        #self.data['llama2_attention_mask'] = llama2_feat['attention_mask']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        age = float(data['age'])
        sex = float(sex_mapping[data['sex']])
        mmse = float(data['mmse'])
        label = dx_mapping[data['dx']]
        lang = lang_mapping[data['language']]
        disvoice_features = [float(data[i+'_mean']) for i in disvoice_feature_names]

        return {
            'input_features': data['input_features'],
            'metadata': [age, sex],
            # 'mmse': mmse,
            'label': label if self.args.task == 'cls' else mmse,
            'lang': lang,
            'disvoice_features': disvoice_features,
            'text_input_ids': data['text_input_ids'],
            'text_attention_mask': data['text_attention_mask'],
            'llama2_input_ids': data.get('llama2_input_ids', None),
            'llama2_attention_mask': data.get('llama2_attention_mask', None),
        }
        
class TAUKADIALTestDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        files = os.listdir('/content/drive/MyDrive/TAUKADIAL-24/test') #/data/datasets/TAUKADIAL-24/test/TAUKADIAL-24/test
        files = [i for i in files if i.endswith('.wav')]

        self.data = pd.read_parquet(f'/content/drive/MyDrive/TAUKADIAL-24/feats_test.parquet') #/data/datasets/TAUKADIAL-24/feature/feats_test.parquet
        for i in self.data.columns:
            if i != 'filename':
                self.data[i+'_mean'] = self.data[i].apply(np.mean).fillna(0)

        self.data['audio'] = [
            load_wave(
                f'/content/drive/MyDrive/TAUKADIAL-24/test/{i}', #/data/datasets/TAUKADIAL-24/test/TAUKADIAL-24/test/{i}
                sample_rate=self.args.sample_rate).flatten() 
            for i in self.data.filename]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.args.method)

        if 'whisper' in self.args.method:
            self.data['input_features'] = [
                self.feature_extractor(i, sampling_rate=self.args.sample_rate, return_tensors="pt")['input_features'][0]
                for i in self.data['audio']
            ]
        
        elif 'wav2vec2' in self.args.method:
            self.data['input_features'] = [
                self.feature_extractor(
                    i,
                    max_length=self.args.max_length,
                    sampling_rate=self.args.sampling_rate,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")['input_values'][0]
                for i in self.data['audio']
            ]
        
        self.en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # Transcribed text
        text = pd.read_parquet('/content/drive/MyDrive/TAUKADIAL-24/translation_test.parquet') #/data/datasets/TAUKADIAL-24/transcription/translation_test.parquet
        self.data = pd.merge(self.data, text, left_on='filename', right_on='file_name')

        text_feat = [
            self.en_tokenizer(j, padding='max_length', max_length=256, truncation=True) if i == 'en' else
            self.cn_tokenizer(j, padding='max_length', max_length=256, truncation=True)
            for i, j in zip(self.data.language, self.data.transcribed_text)
        ]
        self.data['text_input_ids'] = [i['input_ids'] for i in text_feat]
        self.data['text_attention_mask'] = [i['attention_mask'] for i in text_feat]

        # LLAMA-2 explainations
        #self.data['pid'] = self.data.filename.apply(lambda x: x.split('-')[1])
        #self.data['llama2'] = self.data.pid.apply(lambda x: open(f'/data/datasets/TAUKADIAL-24/llama2/llama2_test/{x}.txt').read())

        #llama2_feat = self.en_tokenizer(self.data['llama2'].tolist(), padding='max_length', max_length=256, truncation=True)
        #self.data['llama2_input_ids'] = llama2_feat['input_ids']
        #self.data['llama2_attention_mask'] = llama2_feat['attention_mask']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        # age = float(data['age'])
        # sex = float(sex_mapping[data['sex']])
        lang = lang_mapping[data['language']]
        disvoice_features = [float(data[i+'_mean']) for i in disvoice_feature_names]

        return {
            'input_features': data['input_features'],
            # 'metadata': [age, sex],
            # 'mmse': mmse,
            'label': label,
            'lang': lang,
            'disvoice_features': disvoice_features,
            'text_input_ids': data['text_input_ids'],
            'text_attention_mask': data['text_attention_mask'],
            #'llama2_input_ids': data['llama2_input_ids'],
            #'llama2_attention_mask': data['llama2_attention_mask'],
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: None

    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch
