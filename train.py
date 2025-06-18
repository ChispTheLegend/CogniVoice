import os
import sys
import json
import torch
import wandb
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
    Trainer
)

from cognivoice.model import Whisper, WhisperPoe
from cognivoice.data_processor import *
from cognivoice.training_args import AudioTrainingArguments, RemainArgHfArgumentParser


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# 6.2 Cleanup Function
import shutil
import glob
def cleanup_checkpoints(output_dir, keep_final=True):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for ckpt in checkpoints:
        print(f"Removing checkpoint: {ckpt}")
        shutil.rmtree(ckpt)

    if not keep_final:
        final_model = os.path.join(output_dir, f"final_model_{name}.pt")
        if os.path.exists(final_model):
            os.remove(final_model)
            print(f"Removed final model: {final_model}")

# 6.2 Batch Size Finder
def find_max_batch_size(model, dataset, start=4, max_batch=128):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = start
    while batch_size <= max_batch:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    model(**batch)
            print(f"✅ Batch size {batch_size} fits.")
            batch_size *= 2
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ Batch size {batch_size} caused OOM.")
                break
            else:
                raise e

def main(args):
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    '''
    parser = RemainArgHfArgumentParser((AudioTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file=os.path.abspath(sys.argv[1])
        args, _ = parser.parse_json_file(json_file, return_remaining_args=True) #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        print("ELSE")
        print(sys.argv)
        args = parser.parse_args_into_dataclasses()[0]
    args.dataloader_num_workers = 8
    '''

    # Wandb
    args.report_to = ['wandb']
    project = 'TAUKADIAL-2025'
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
    run_name = name + '-' + str(args.seed) if not args.wandb_run_id else name # Name for W&B display #name += '-' + str(args.seed)

    # 6.17 initialize, core change for Wandb run resumption
    wandb_resume_mode = "allow" # Default to 'allow' for flexibility
    if args.wandb_run_id:
        # If a specific run ID is provided, assume we *must* resume it
        wandb_resume_mode = "must"
        logger.info(f"Attempting to resume W&B run with ID: {args.wandb_run_id}, mode: '{wandb_resume_mode}'")
    else:
        logger.info(f"Initializing a new W&B run (or allowing resume if ID {run_name} exists), mode: '{wandb_resume_mode}'")

    wandb.init(
        project=project,
        group=group,
        name=run_name, # This name is primarily for display in the W&B UI
        config=args,
        id=args.wandb_run_id if args.wandb_run_id else None, # Pass the explicit ID if provided
        resume=wandb_resume_mode,
        tags=["final", args.task]
    )
    set_seed(args.seed)

    # Format HF model names
    if args.method == 'wav2vec':
        args.method = 'facebook/' + args.method
    
    elif args.method.startswith('whisper'):
        args.method = 'openai/' + args.method

    from sklearn.model_selection import StratifiedKFold
    data = pd.read_csv('/content/drive/MyDrive/TAUKADIAL-24/train/groundtruth.csv')
    
    label_col = 'dx' if args.task == 'cls' else 'mmse'
    args.metric_for_best_model = 'f1' if args.task == 'cls' else 'mse'
    args.greater_is_better = True if args.task == 'cls' else False
    kv = StratifiedKFold(n_splits=args.num_fold)

    # pred_data = TAUKADIALTestDataset(args)

    scores = []
    for fold_id, (train_idx, eval_idx) in enumerate(tqdm(kv.split(data.drop(label_col, axis=1), data[label_col]), desc='Cross Validation')):
        args.output_dir = os.path.join(output_dir_root, f'fold_{fold_id}')
        
        # Dataset
        train_data = TAUKADIALDataset(args, subset=train_idx)
        eval_data = TAUKADIALDataset(args, subset=eval_idx)

        # Model
        if args.method == 'wav2vec':
            model = AutoModelForSequenceClassification.from_pretrained(args.method)
        
        elif 'whisper' in args.method:
            if args.use_poe:
                model = WhisperPoe(args)
            else:
                model = Whisper(args)
                # model.parallelize()
        else:
            raise NotImplementedError

        # 6.17.25 LOAD WEIGHTS FROM WANDB ARTIFACT IF SPECIFIED (pre-training initialization) ---
        # This is for warm-starting from a *specific model checkpoint artifact*, which might be from a different run.
        if args.resume_from_wandb_artifact:
            logger.info(f"Downloading model from W&B artifact: {args.resume_from_wandb_artifact}")
            try:
                artifact = wandb.use_artifact(args.resume_from_wandb_artifact, type='model')
                artifact_dir = artifact.download()
                
                # Determine the model file path within the downloaded artifact directory
                # If your artifact is just the .pt file:
                model_file_in_artifact = os.path.join(artifact_dir, f"final_model_{base_name.split('-')[0]}.pt") # Adjust based on how you saved
                
                # If the artifact contains a full Hugging Face Trainer checkpoint structure (e.g., a 'checkpoint-XYZ' folder with pytorch_model.bin)
                # You might need to change this to load from the artifact_dir directly if it's an HF checkpoint.
                # Example: If artifact_dir is '/wandb/artifacts/model-v0' and it contains 'checkpoint-100/pytorch_model.bin'
                # then model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifact_dir, 'checkpoint-100'))
                
                if os.path.exists(model_file_in_artifact):
                    loaded_state_dict = torch.load(model_file_in_artifact, map_location='cpu')
                    model.load_state_dict(loaded_state_dict, strict=False)
                    logger.info(f"Successfully loaded weights from {model_file_in_artifact}")
                else:
                    # Fallback for HF Trainer checkpoints
                    logger.warning(f"Did not find expected .pt file: {model_file_in_artifact}. Attempting to load as HF Trainer checkpoint.")
                    try:
                        # Find a checkpoint folder if the artifact is a full HF Trainer save
                        checkpoint_folders = [d for d in os.listdir(artifact_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(artifact_dir, d))]
                        if checkpoint_folders:
                            latest_checkpoint_folder = os.path.join(artifact_dir, sorted(checkpoint_folders, key=lambda x: int(x.split('-')[1]))[-1])
                            logger.info(f"Loading HF model from checkpoint folder: {latest_checkpoint_folder}")
                            # For HF models, load using from_pretrained from the folder
                            if args.method == 'wav2vec':
                                model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint_folder)
                            elif 'whisper' in args.method:
                                if args.use_poe:
                                    # WhisperPoe needs to be re-initialized and then state_dict loaded.
                                    # This is more complex if WhisperPoe is not directly compatible with HF from_pretrained.
                                    # For now, let's assume if it's a full HF checkpoint, the model type is simple.
                                    # You'd likely load the base Whisper part and then rebuild Poe.
                                    logger.error("Loading WhisperPoe from full HF checkpoint folder is more complex, requires specific WhisperPoe loading logic.")
                                    sys.exit(1)
                                else:
                                    # For Whisper, you'd typically load the HF model and processor
                                    # from_pretrained the checkpoint folder.
                                    model = Whisper.from_pretrained(latest_checkpoint_folder) # Assuming Whisper class has from_pretrained
                                    # Note: Your Whisper class might not have a .from_pretrained method.
                                    # If not, you'd load the base HF model first, then load your custom model's state_dict.
                                    # E.g., base_model = AutoModelForSpeechSeq2Seq.from_pretrained(latest_checkpoint_folder)
                                    #       model = Whisper(args) # Re-initialize your custom model
                                    #       model.model.load_state_dict(base_model.state_dict()) # Load into the base part
                            logger.info(f"Successfully loaded model from Hugging Face checkpoint in artifact.")
                        else:
                            raise FileNotFoundError(f"No .pt file or checkpoint folder found in artifact {artifact_dir}")
                    except Exception as inner_e:
                        logger.error(f"Failed to load as HF Trainer checkpoint: {inner_e}")
                        sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load model weights from W&B artifact {args.resume_from_wandb_artifact}: {e}")
                sys.exit(1)
        # --- END WANDB ARTIFACT LOADING ---
        
        metric = load_metric("./cognivoice/metrics.py", args.task, trust_remote_code=True) #metric = load_metric("./cognivoice/metrics.py", args.task)

        def new_compute_metrics(results):
            labels, label_mmse, sex_labels, lng_labels, pic_labels = results.label_ids
            if isinstance(results.predictions, tuple):
                logits, mmse_pred = results.predictions
            else:
                logits = results.predictions
                mmse_pred = None
            predictions = logits.argmax(-1)
            metrics = {}
            metrics['uar'] = recall_score(labels, predictions, average='macro')
            metrics['f1'] = f1_score(labels, predictions, average='binary')
            # metrics['f1'] = f1_score(labels, predictions)
            metrics['mse']  = mean_squared_error(label_mmse, mmse_pred)
            metrics['r2']  = r2_score(label_mmse, mmse_pred)
            for sex in set(sex_labels):
                sub_labels = labels[sex_labels == sex]
                sub_predictions = predictions[sex_labels == sex]
                sub_labels_mmse = label_mmse[sex_labels == sex]
                sub_predictions_mmse = mmse_pred[sex_labels == sex]
                metrics['mse_%s'%sex_map_rev[sex]] = mean_squared_error(sub_labels_mmse, sub_predictions_mmse)
                metrics['r2_%s'%sex_map_rev[sex]] = r2_score(sub_labels_mmse, sub_predictions_mmse)
            for lng in set(lng_labels):
                sub_labels = labels[lng_labels == lng]
                sub_predictions = predictions[lng_labels == lng]
                sub_labels_mmse = label_mmse[sex_labels == sex]
                sub_predictions_mmse = mmse_pred[sex_labels == sex]
                metrics['uar_%s'%lng_map_rev[lng]] = recall_score(sub_labels, sub_predictions, average='macro')
                metrics['f1_%s'%lng_map_rev[lng]] = f1_score(sub_labels, sub_predictions, average='binary')
                metrics['mse_%s'%lng_map_rev[lng]] = mean_squared_error(sub_labels_mmse, sub_predictions_mmse)
                metrics['r2_%s'%lng_map_rev[lng]] = r2_score(sub_labels_mmse, sub_predictions_mmse)
            return metrics

        def compute_metrics(p: EvalPrediction):
            # labels, label_mmse, sex_labels, lng_labels, pic_labels = p.label_ids
            # breakpoint()
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1) if args.task == 'cls' else np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            result["combined_score"] = np.mean(list(result.values())).item()

            return result

        # Train
        # args.max_steps = 3
        # args.eval_steps = 1
        data_collator = default_data_collator if args.pad_to_max_length else None
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
            # tokenizer=tokenizer,
            # data_collator=data_collator,
        )

        # args.overwrite_output_dir = False
        # args.do_train = False
        # model.load_state_dict(ckpt)
        if args.do_train:
            # Detecting last checkpoint.
            last_checkpoint = None
            if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(args.output_dir)
                if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
                    raise ValueError(
                        f"Output directory ({args.output_dir}) already exists and is not empty. "
                        "Use --overwrite_output_dir to overcome."
                    )
                elif last_checkpoint is not None and args.resume_from_checkpoint is None:
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                    )
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            metrics = train_result.metrics

            trainer.save_model()  # Saves the tokenizer too for easy upload
    
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if args.do_eval:
            logger.info("*** Evaluate ***")
            key = 'eval_' + args.metric_for_best_model
            eval_name = f'fold_{fold_id}'
            metrics = trainer.evaluate(eval_dataset=eval_data)
            trainer.log_metrics(eval_name, metrics)
            trainer.save_metrics(eval_name, metrics)

            scores.append(metrics[key])

            logger.info('*** Predict ***')
            predictions = trainer.predict(eval_data, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1) if args.task == 'cls' else np.squeeze(predictions)
            
            output_predict_file = os.path.join(output_dir_root, f"pred_{args.task}_fold_{fold_id}.csv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("idx,pred\n")
                    for i, p in zip(eval_idx, predictions):
                        writer.write(f'{i},{p}\n')

    wandb_log = {}
    for i, j in enumerate(scores):
        print(f'Fold: {i}, Score: {j:.4f}')
        wandb_log[f'per_fold/fold_{i}'] = j
    print('Mean score:', f'{np.mean(scores):.2f}')
    wandb_log['mean'] = np.mean(scores)
    wandb_log['std'] = np.std(scores)
    wandb_log['max'] = np.max(scores)
    wandb_log['min'] = np.min(scores)

    with open(os.path.join(output_dir_root, 'result.json'), 'w') as f:
        json.dump(wandb_log, f, indent=4)
    
    wandb.log(wandb_log)

    #6.2.25 Save final weights explicitly 
    final_path = os.path.join(args.output_dir, f"final_model_{name}.pt")
    torch.save(model.state_dict(), final_path)
    # Upload to wandb
    artifact = wandb.Artifact(f"{name}-final-model", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()
