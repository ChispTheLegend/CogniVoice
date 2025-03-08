print("train.py started")

import os
import sys
import json
import torch
import logging
import numpy as np
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
    Trainer,
)

from cognivoice.model import Whisper, WhisperPoe
from cognivoice.data_processor import *
from cognivoice.training_args import AudioTrainingArguments, RemainArgHfArgumentParser

# Import Neptune
import neptune
from neptune.types import File
from google.colab import userdata

#new imports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Environment variables: {os.environ}")
neptune_api_token = os.environ.get('NEPTUNE_API_TOKEN')
logging.info(f"neptune_api_token: {neptune_api_token}")


print("all imports worked")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():
    parser = RemainArgHfArgumentParser((AudioTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file = os.path.abspath(sys.argv[1])
        args, _ = parser.parse_json_file(json_file, return_remaining_args=True)
    else:
        args = parser.parse_args_into_dataclasses()[0]
    args.dataloader_num_workers = 8

    project = "TAUKADIAL-2024"
    group = args.task
    name = args.method
    output_dir_root = args.output_dir
    if args.use_disvoice:
        name += "_disvoice"
        output_dir_root += "_disvoice"
    if args.use_metadata:
        name += "_metadata"
        output_dir_root += "_metadata"
    if args.use_text:
        name += "_text"
        output_dir_root += "_text"
    if args.use_llama2:
        name += "_llama2"
        output_dir_root += "_llama2"
    if args.use_poe:
        name += "_poe"
        output_dir_root += "_poe"
    name += "-" + str(args.seed)

    set_seed(args.seed)

    if args.method == "wav2vec":
        args.method = "facebook/" + args.method
    elif args.method.startswith("whisper"):
        args.method = "openai/" + args.method

    from sklearn.model_selection import StratifiedKFold

    data = pd.read_csv("/content/drive/MyDrive/TAUKADIAL-24/train/groundtruth.csv")
    data = data[:20]
    label_col = "dx" if args.task == "cls" else "mmse"
    args.metric_for_best_model = "f1" if args.task == "cls" else "mse"
    args.greater_is_better = True if args.task == "cls" else False
    kv = StratifiedKFold(n_splits=args.num_fold)

    # Initialize Neptune
    try:
        #api_token = userdata.get("NEPTUNE_API_TOKEN")
        #print(f"Api token is: {api_token}") #add this line to see the api token
        run = neptune.init_run(
            project="chispen-workspace/TAUKADIAL2025",  # Replace with your workspace/project
            name=name,
            tags=[args.task, args.method],
            #api_token=userdata.get('NEPTUNE_API_TOKEN'),
            api_token=neptune_api_token
        )
        run["parameters"] = vars(args)  # Log hyperparameters
    except KeyError:
        print("Error: Neptune API token not found in Colab Secrets.")
        return
    except Exception as e:
        print(f"An error occurred while initializing Neptune: {e}")
        return

    scores = []
    for fold_id, (train_idx, eval_idx) in enumerate(
        tqdm(
            kv.split(data.drop(label_col, axis=1), data[label_col]),
            desc="Cross Validation",
        )
    ):
        args.output_dir = os.path.join(output_dir_root, f"fold_{fold_id}")

        train_data = TAUKADIALDataset(args, subset=train_idx)
        eval_data = TAUKADIALDataset(args, subset=eval_idx)

        if args.method == "wav2vec":
            model = AutoModelForSequenceClassification.from_pretrained(args.method)
        elif "whisper" in args.method:
            if args.use_poe:
                model = WhisperPoe(args)
            else:
                model = Whisper(args)
        else:
            raise NotImplementedError

        metric = load_metric("./cognivoice/metrics.py", args.task)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1) if args.task == "cls" else np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            result["combined_score"] = np.mean(list(result.values())).item()
            return result

        data_collator = default_data_collator if args.pad_to_max_length else None
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

            run[f"model/fold_{fold_id}/checkpoint"].upload(
                File.as_artifact(os.path.join(args.output_dir, "pytorch_model.bin"))
            )

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            for key, value in metrics.items():
                run[f"train/{key}"] = value

        if args.do_eval:
            logger.info("*** Evaluate ***")
            key = "eval_" + args.metric_for_best_model
            eval_name = f"fold_{fold_id}"
            metrics = trainer.evaluate(eval_dataset=eval_data)
            trainer.log_metrics(eval_name, metrics)
            trainer.save_metrics(eval_name, metrics)

            scores.append(metrics[key])

            for key, value in metrics.items():
                run[f"eval/{eval_name}/{key}"] = value

            logger.info("*** Predict ***")
            predictions = trainer.predict(eval_data, metric_key_prefix="predict").predictions
            predictions = (
                np.argmax(predictions, axis=1) if args.task == "cls" else np.squeeze(predictions)
            )

            output_predict_file = os.path.join(
                output_dir_root, f"pred_{args.task}_fold_{fold_id}.csv"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("idx,pred\n")
                    for i, p in zip(eval_idx, predictions):
                        writer.write(f"{i},{p}\n")

    if trainer.is_world_process_zero():
        run.stop()

if __name__ == "__main__":
    main()