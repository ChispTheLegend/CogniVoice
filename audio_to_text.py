import transformers
from transformers import pipeline
from glob import glob
import pandas as pd
import os
from tqdm import tqdm
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

whisper = pipeline('automatic-speech-recognition', model = "openai/whisper-large-v3", device = 1, return_timestamps=True)

train_audio_loc = "./data/taukadial/train/*.wav"
train_audio_files = glob(train_audio_loc)


audio_to_text = {}
for f in tqdm(train_audio_files):
    output = whisper(f)
    transcribed_text = output["text"]
    doc = nlp(transcribed_text)
    language = doc._.language['language']
    print(language)
    # language = "en" if language == "en" else "cn"
    record = {}
    record['file_name'] = os.path.basename(f).rstrip('.mp3')
    record['transcribed_text'] = transcribed_text
    record['language'] = language

    audio_to_text[len(audio_to_text)] = record


df = pd.DataFrame(audio_to_text)
df = df.T
df.to_parquet("./data/taukadial/text/audio_to_text_w_language_train.parquet")
