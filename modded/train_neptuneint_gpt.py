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
    Trainer
)
import neptune.new as neptune

# Your code...

def main():
    run = neptune.init(
        project='your_workspace/your_project',  # replace with your Neptune workspace and project name
        api_token='your_api_token'
    )
    
    run['config'] = {
        'project': project,
        'group': group,
        'name': name,
        'output_dir_root': output_dir_root,
        'method': args.method,
        'use_disvoice': args.use_disvoice,
        'use_metadata': args.use_metadata,
        'use_text': args.use_text,
        'use_llama2': args.use_llama2,
        'use_poe': args.use_poe,
        'seed': args.seed
    }

    # The rest of your main function...
    
    if args.do_train:
        # Training code...
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        metrics = train_result.metrics

        # Log metrics to Neptune
        run['train/metrics'] = metrics

        trainer.save_model()

    if args.do_eval:
        # Evaluation code...
        metrics = trainer.evaluate(eval_dataset=eval_data)
        
        # Log eval metrics to Neptune
        run[f'eval/{eval_name}/metrics'] = metrics

    # Log final results
    run['final_scores'] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'max': np.max(scores),
        'min': np.min(scores)
    }
    
    # Finish Neptune run
    run.stop()

if __name__ == "__main__":
    main()
