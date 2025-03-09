import os
import sys
import json
import torch
import logging
import numpy as np
import neptune
  # Import Neptune
from tqdm import tqdm
from datasets import load_metric
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    set_seed,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
    default_data_collator,
    WhisperFeatureExtractor, 
    WhisperTokenizer,
    WhisperProcessor,
    AutoModelForSequenceClassification,
    Trainer
)

from cognivoice.model import Whisper, WhisperPoe
from cognivoice.data_processor import *
from cognivoice.training_args import AudioTrainingArguments, RemainArgHfArgumentParser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    parser = RemainArgHfArgumentParser((AudioTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file = os.path.abspath(sys.argv[1])
        args, _ = parser.parse_json_file(json_file, return_remaining_args=True)
    else:
        args = parser.parse_args_into_dataclasses()[0]
    args.dataloader_num_workers = 8

    project = 'TAUKADIAL-2024'
    group = args.task
    name = args.method
    output_dir_root = args.output_dir

    if args.use_disvoice:
        name += '_disvoice'
        output_dir_root += '_disvoice'
    if args.use_metadata:
        name += '_metadata'
        output_dir_root += '_metadata'
    if args.use_text:
        name += '_text'
        output_dir_root += '_text'
    if args.use_llama2:
        name += '_llama2'
        output_dir_root += '_llama2'
    if args.use_poe:
        name += '_poe'
        output_dir_root += '_poe'
    name += '-' + str(args.seed)

    # Initialize Neptune
    run = neptune.init_run(
        project="chispen-workspace/TAUKADIAL2025",
        name=name,
        tags=[args.task, args.method],
        api_token=os.environ.get("NEPTUNE_API_TOKEN")  # Replace or set via environment variable
    )
    run["parameters"] = vars(args)  # Log hyperparameters

    set_seed(args.seed)

    from sklearn.model_selection import StratifiedKFold
    data = pd.read_csv('/content/drive/MyDrive/TAUKADIAL-24/train/groundtruth.csv')
    data = data[:20]
    label_col = 'dx' if args.task == 'cls' else 'mmse'
    args.metric_for_best_model = 'f1' if args.task == 'cls' else 'mse'
    args.greater_is_better = True if args.task == 'cls' else False
    kv = StratifiedKFold(n_splits=args.num_fold)

    scores = []
    for fold_id, (train_idx, eval_idx) in enumerate(tqdm(kv.split(data.drop(label_col, axis=1), data[label_col]), desc='Cross Validation')):
        args.output_dir = os.path.join(output_dir_root, f'fold_{fold_id}')

        train_data = TAUKADIALDataset(args, subset=train_idx)
        eval_data = TAUKADIALDataset(args, subset=eval_idx)

        if args.method == 'wav2vec':
            model = AutoModelForSequenceClassification.from_pretrained(args.method)
        elif 'whisper' in args.method:
            model = WhisperPoe(args) if args.use_poe else Whisper(args)
        else:
            raise NotImplementedError

        metric = load_metric("./cognivoice/metrics.py", args.task)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1) if args.task == 'cls' else np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            result["combined_score"] = np.mean(list(result.values())).item()
            return result

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
        )

        if args.do_train:
            last_checkpoint = None
            if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(args.output_dir)

            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            metrics = train_result.metrics

            trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            # Log training metrics to Neptune
            for key, value in metrics.items():
                run[f"train/{key}"] = value

        if args.do_eval:
            logger.info("*** Evaluate ***")
            key = 'eval_' + args.metric_for_best_model
            eval_name = f'fold_{fold_id}'
            metrics = trainer.evaluate(eval_dataset=eval_data)

            trainer.log_metrics(eval_name, metrics)
            trainer.save_metrics(eval_name, metrics)

            scores.append(metrics[key])

            for key, value in metrics.items():
                run[f"eval/{eval_name}/{key}"] = value

            logger.info('*** Predict ***')
            predictions = trainer.predict(eval_data, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1) if args.task == 'cls' else np.sq
