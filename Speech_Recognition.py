# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pip install librosa pydub

%env JOBLIB_TEMP_FOLDER=/tmp


# !rm -r /kaggle/working/train_wav

!ls /kaggle/input/bengaliai-speech/examples 

# !ls /kaggle/working/train_wav

df_train_val=pd.read_csv('/kaggle/input/bengaliai-speech/train.csv')

df_train_val

df_train_val.columns

df_train=df_train_val.loc[df_train_val['split']=='train']

df_train_val.split.value_counts()

df_val=df_train_val.loc[df_train_val['split']=='valid']

df_val=df_val[['id', 'sentence']]

df_train=df_train[['id', 'sentence']]

###################################

mkdir /kaggle/working/train_test1

!cp /kaggle/input/bengaliai-speech/train_mp3s/000005f3362c.mp3 /kaggle/working/batch_train_files
!cp /kaggle/input/bengaliai-speech/train_mp3s/00001dddd002.mp3 /kaggle/working/batch_train_files
!cp /kaggle/input/bengaliai-speech/train_mp3s/00001e0bc131.mp3 /kaggle/working/batch_train_files

# mp3_directory = '/kaggle/working/train_test1'

# for filename in os.listdir(mp3_directory):
#     print(filename)

%%time
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
from scipy.signal import resample_poly
import numpy as np

def convert_mp3_to_wav(mp3_path, wav_path, target_sample_rate=16000):
    # Load MP3 file using pydub
    audio = AudioSegment.from_mp3(mp3_path)
    
    # Export as WAV
    audio.export("temp.wav", format="wav")

    # Load the temporary WAV file with librosa
    signal, sample_rate = librosa.load("temp.wav", sr=None)

    # Calculate resampling up and down factors
    lcm = np.lcm(target_sample_rate, sample_rate)
    up = lcm // sample_rate
    down = lcm // target_sample_rate

    # Resample to the target sample rate
    signal_resampled = resample_poly(signal, up, down)

    # Save the resampled WAV file
    sf.write(wav_path, signal_resampled, target_sample_rate)

    # Remove the temporary file
    os.remove("temp.wav")

# Path to the directory containing MP3 files
mp3_directory = '/kaggle/input/bengaliai-speech/train_mp3s'

# Path to the directory where you want to save the WAV files
wav_directory = '/kaggle/working/train_wav/'

# Create the WAV directory if it doesn't exist
if not os.path.exists(wav_directory):
    os.makedirs(wav_directory)

# Iterate over all MP3 files in the directory
# for filename in os.listdir(mp3_directory):
#     if filename.endswith('.mp3'):
#         mp3_path = os.path.join(mp3_directory, filename)
#         wav_filename = os.path.splitext(filename)[0] + '.wav'
#         wav_path = os.path.join(wav_directory, wav_filename)
counter=0
for filename in os.listdir(mp3_directory):
    if filename.endswith('.mp3'):
#         # Increment the counter
        counter += 1

#         # If counter is greater than 10, exit the loop
#         if counter > 10:
#             break

        mp3_path = os.path.join(mp3_directory, filename)
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(wav_directory, wav_filename)

#         print(f"Converting {filename} to WAV format...")
        convert_mp3_to_wav(mp3_path, wav_path)

print("Conversion complete!")
print(counter)

# Path to the directory containing MP3 files
# mp3_directory = '/kaggle/input/bengaliai-speech/train_mp3s'

# Path to the directory where you want to save the WAV files
# wav_directory = '/kaggle/working/test_wav/'

# Create the WAV directory if it doesn't exist
if not os.path.exists(wav_directory):
    os.makedirs(wav_directory)

# Iterate over all MP3 files in the directory
# for filename in os.listdir(mp3_directory):
#     if filename.endswith('.mp3'):
#         mp3_path = os.path.join(mp3_directory, filename)
#         wav_filename = os.path.splitext(filename)[0] + '.wav'
#         wav_path = os.path.join(wav_directory, wav_filename)
# for filename in os.listdir(mp3_directory):
#     if filename.endswith('.mp3'):
#         mp3_path = os.path.join(mp3_directory, filename)
#         wav_filename = os.path.splitext(filename)[0] + '.wav'
#         wav_path = os.path.join(wav_directory, wav_filename)
#         print(f"Converting {filename} to WAV format...")
#         convert_mp3_to_wav(mp3_path, wav_path)

###################################

%%time
import pandas as pd
import os
from datasets import Dataset, Features, Audio, Value, DatasetDict
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# Path to CSV file
csv_path = '/kaggle/input/bengaliai-speech/train.csv'
# Path to audio files
audio_directory = '/kaggle/input/bengaliai-speech/train_mp3s'

# Read the CSV file
data = pd.read_csv(csv_path)

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

def prepare_sample(row):
    audio_id = row['id']
    sentence = row['sentence']
    audio_path = os.path.join(audio_directory, audio_id + '.mp3')
    if os.path.exists(audio_path):
        return {
            'audio': audio_path,
            'sentence': sentence
        }
    return None

def prepare_dataset(data):
    with Pool(processes=4) as pool:  # Adjust the number of processes as needed
        samples = pool.map(prepare_sample, data.to_dict(orient='records'))
        samples = [sample for sample in samples if sample is not None]

    features = Features({
        'audio': Audio(sampling_rate=16000, mono=True),
        'sentence': Value(dtype='string'),
    })

    return Dataset.from_dict({'audio': [s['audio'] for s in samples], 'sentence': [s['sentence'] for s in samples]}, features=features)

# Prepare training and testing datasets
train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
})


# import pandas as pd
# import os
# from datasets import Dataset, Features, Audio, Value, DatasetDict
# from sklearn.model_selection import train_test_split

# # Path to CSV file
# csv_path = '/kaggle/input/bengaliai-speech/train.csv'
# # Path to audio files
# audio_directory = '/kaggle/input/bengaliai-speech/train_mp3s'

# # Read the CSV file
# data = pd.read_csv(csv_path)

# # Split data into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2)

# def prepare_dataset(data):
#     audio_data = []
#     transcriptions = []

#     for index, row in data.iterrows():
#         audio_id = row['id']
#         sentence = row['sentence']
#         audio_path = os.path.join(audio_directory, audio_id + '.mp3')
#         if os.path.exists(audio_path):
#             audio_data.append(audio_path)
#             transcriptions.append(sentence)

#     aligned_data = {
#         'audio': audio_data,
#         'sentence': transcriptions
#     }

#     features = Features({
#         'audio': Audio(sampling_rate=16000, mono=True),
#         'sentence': Value(dtype='string'),
#     })

#     return Dataset.from_dict(aligned_data, features=features)

# # Prepare training and testing datasets
# train_dataset = prepare_dataset(train_data)
# test_dataset = prepare_dataset(test_data)

# # Combine into a DatasetDict
# dataset_dict = DatasetDict({
#     'train': train_dataset,
#     'test': test_dataset,
# })

from datasets import load_dataset
dataset_dict.save_to_disk("/kaggle/working/dataset_dict.hf")

####################################

!pip install transformers[torch]
!pip install datasets

# from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

# TO_LANGUAGE_CODE

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="bengali", task="transcribe"
)

from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
dataset = dataset_dict.cast_column("audio", Audio(sampling_rate=sampling_rate))

sampling_rate

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example



# dataset = dataset.map(
#     prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2
# )

# import os
# path = '/kaggle/input/bengaliai-speech/train_mp3s/108c4c412b68.mp3'

# os.path.exists(path)

max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length

# dataset = dataset.filter(
#     is_audio_in_length_range,
#     input_columns=["input_length"],
# )

#################################################

dataset

from IPython.display import Audio
audio = Audio(filename='/kaggle/input/bengaliai-speech/train_mp3s/8dfee904c654.mp3')
display(audio)



print(dataset['train'].features)
print(dataset['test'].features)


import os

file_path = '/kaggle/input/bengaliai-speech/train_mp3s/8dfee904c654.mp3'
print(os.path.exists(file_path))  # should print True if the file exists


pip install torchaudio



# # Example: Checking the first 5 file paths in the training set
# for i in range(5):
#     print(dataset['train']['audio'][i])

# dataset['train']['audio']

from datasets import load_dataset
dataset.save_to_disk("/kaggle/working/dataset.hf")



import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

########################

!pip install evaluate
!pip install jiwer

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

# import jiwer  # you may need to install this library

# def mean_wer(solution, submission):
#     joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
#     domain_scores = joined.groupby('domain').apply(
#         # note that jiwer.wer computes a weighted average wer by default when given lists of strings
#         lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
#     )
#     return domain_scores.mean()

# assert (solution.columns == ['id', 'domain', 'sentence']).all()
# assert (submission.columns == ['id',' sentence']).all()


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="bengali", task="transcribe", use_cache=True
)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-bengali",  # name on the HF Hub
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()
