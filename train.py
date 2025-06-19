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

    # 6.18 Wandb resume logic
    WANDB_RUN_ID_ENV = os.environ.get("WANDB_RUN_ID")
    resume_id = WANDB_RUN_ID_ENV if WANDB_RUN_ID_ENV else (args.wandb_run_id if hasattr(args, 'wandb_run_id') else None)
    
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
    name += '-' + str(args.seed)

    run = wandb.init(
        project=project,
        group=group,
        name=name, 
        config=args,
        id=resume_id,
        resume="allow",
        tags=["final", args.task]
    )

    args.wandb_run_id = run.id
    logger.info(f"Wandb run initialized with ID: {run.id}, Name: {run.name}")

    set_seed(args.seed)

    # 6.18 Initialize W&B API (Crucial for fetching artifacts outside the current run's direct logs)
    api = wandb.Api()

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
            
            # --- NEW BLOCK FOR W&B ARTIFACT LOADING ---
            last_checkpoint_path = None
            if resume_id:
                logger.info(f"Attempting to restore checkpoint from W&B Artifact for run ID: {resume_id}")
                try:
                    # Find the run we are trying to resume from
                    # Use wandb.run.entity and wandb.run.project for current run context,
                    # or pass them through args if they are consistent.
                    run_to_resume_from = api.run(f"{run.entity}/{run.project}/{resume_id}") # Use the run object initialized earlier
    
                    # Search for checkpoint artifacts logged by this specific run
                    # The artifact name should be consistent for a fold's checkpoints.
                    # Example naming: "model-checkpoint-fold-0-v0", "model-checkpoint-fold-0-v1", etc.
                    # Or you could log a single "latest" alias that gets updated.
    
                    # Let's assume you will log artifacts named "checkpoint-fold_X-step_Y"
                    # We need to find the latest one for the current fold.
                    
                    # A more robust way to get the latest by iterating, as there's no direct ":latest" alias for *all* checkpoints
                    # (only for the overall collection or if you explicitly alias it).
                    
                    # Filter for artifacts of type 'model-checkpoint' relevant to this fold
                    found_artifacts = []
                    for art in run_to_resume_from.logged_artifacts():
                        # Match based on artifact type and a name pattern for the fold
                        if art.type == "model-checkpoint" and f"checkpoint-fold_{fold_id}" in art.name:
                            found_artifacts.append(art)
                    
                    if found_artifacts:
                        # Sort by version (highest version is latest) or by creation time
                        found_artifacts.sort(key=lambda x: x.version, reverse=True) # Assuming v0, v1, v2...
                        latest_checkpoint_artifact = found_artifacts[0]
                        
                        logger.info(f"Found latest checkpoint artifact: {latest_checkpoint_artifact.name}")
                        
                        # Create a temporary directory to download the artifact to
                        # This will ensure it doesn't conflict with other runs/folds
                        temp_download_dir = os.path.join(args.output_dir, f"wandb_checkpoint_download_fold_{fold_id}")
                        os.makedirs(temp_download_dir, exist_ok=True)
    
                        artifact_dir = latest_checkpoint_artifact.download(root=temp_download_dir)
                        logger.info(f"Checkpoint artifact downloaded to: {artifact_dir}")
    
                        # The Trainer's get_last_checkpoint needs to find the actual checkpoint folder
                        # inside the downloaded artifact directory.
                        last_checkpoint_path = get_last_checkpoint(artifact_dir)
                        
                        if last_checkpoint_path is None:
                            logger.warning(f"No valid checkpoint sub-directory found inside downloaded artifact: {artifact_dir}")
                            # If no valid checkpoint is found in artifact, it will start training from scratch.
                    else:
                        logger.info(f"No 'model-checkpoint' artifacts found for run {resume_id} and fold {fold_id}. Starting from scratch.")
    
                except wandb.errors.CommError as e:
                    logger.warning(f"Could not retrieve W&B run or artifacts for ID {resume_id}: {e}. Starting from scratch.")
            # --- END NEW BLOCK FOR W&B ARTIFACT LOADING ---

            #6.18 modded last checkpoint detection
            if last_checkpoint_path is None and os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
              temp_local_checkpoint = get_last_checkpoint(args.output_dir)
            if temp_local_checkpoint:
                last_checkpoint_path = temp_local_checkpoint
                logger.info(f"Found local checkpoint at {last_checkpoint_path}. Using this for resuming.")
            elif len(os.listdir(args.output_dir)) > 0:
                 raise ValueError(
                     f"Output directory ({args.output_dir}) already exists and is not empty. "
                     "Use --overwrite_output_dir to overcome."
                 )
            
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint_path)
            metrics = train_result.metrics

            trainer.save_model()  # Saves the tokenizer too for easy upload

            # --- NEW BLOCK FOR LOGGING CHECKPOINTS AS ARTIFACTS ---
            # Get the path to the latest checkpoint saved by the Trainer
            current_fold_output_dir = os.path.join(output_dir_root, f'fold_{fold_id}') # Ensure this is the actual output dir
            latest_trainer_checkpoint = get_last_checkpoint(current_fold_output_dir)
            
            if latest_trainer_checkpoint and trainer.is_world_process_zero():
                # Create a unique name for the artifact (e.g., including step and fold_id)
                artifact_name = f"checkpoint-fold_{fold_id}-step_{wandb.run.step}"
                checkpoint_artifact = wandb.Artifact(
                    artifact_name,
                    type="model-checkpoint",
                    description=f"Trainer checkpoint for fold {fold_id} at step {wandb.run.step}"
                )
                checkpoint_artifact.add_dir(latest_trainer_checkpoint) # Add the entire checkpoint directory
                wandb.log_artifact(checkpoint_artifact, aliases=["latest-fold-checkpoint", f"latest-fold-{fold_id}"]) # Add aliases
                logger.info(f"Logged checkpoint artifact: {checkpoint_artifact.name}")
            # --- END NEW BLOCK FOR LOGGING CHECKPOINTS AS ARTIFACTS ---

            
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
