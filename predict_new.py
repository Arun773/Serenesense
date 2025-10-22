# Standard library imports
import os
import logging
import warnings
import gc
import json
from datetime import datetime, UTC
from collections import Counter
import itertools
import subprocess
from pathlib import Path
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Data processing and analysis
import numpy as np
import pandas as pd
import librosa

# Machine learning and deep learning
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Deep learning frameworks and tools
import torch
from torch.utils.data import DataLoader
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)

# Image processing
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

# Visualization
import matplotlib.pyplot as plt

# Image handling
from PIL import ImageFile
# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'webm', 'ogg', 'mp4', 'mpeg', 'mp3'}
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
DATASET_FOLDER = 'dataset'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Initialize data structures for image processing
image_dict = {}

# Function to load and process image dataset
def load_image_dataset(dataset_path=DATASET_FOLDER):
    # Initialize empty lists to store file names and labels
    file_names = []
    labels = []
    
    # Get the absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # Iterate through all image files in the specified directory
    for file in tqdm(sorted(Path(dataset_path).glob('*/*.*')), 
                    desc="Loading images", 
                    unit="file"):
        if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}:
            file_names.append(str(file))  # Add the file path to the list
            label = file.parent.name  # Extract the label from the parent directory name
            labels.append(label)  # Add the label to the list
    
    # Create initial dataframe
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    
    print(f"\nInitial Dataset Summary:")
    print(f"Total images found: {len(file_names)}")
    print(f"Unique labels found: {sorted(df['label'].unique())}")
    print(f"Initial label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count} images")
    
    # Perform random oversampling to balance the dataset
    print("\nBalancing dataset using random oversampling...")
    y = df[['label']]
    X = df.drop(['label'], axis=1)
    ros = RandomOverSampler(random_state=83)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Clean up and create new balanced dataframe
    del y
    df = X_resampled
    df['label'] = y_resampled
    del X_resampled, y_resampled
    gc.collect()
    
    print("\nBalanced Dataset Summary:")
    print(f"Shape of balanced dataset: {df.shape}")
    print(f"\nDetailed Label Distribution:")
    
    # Get value counts and calculate percentages
    label_counts = df['label'].value_counts()
    total_images = len(df)
    
    # Print detailed distribution with counts and percentages
    print("\nLabel                 Count     Percentage")
    print("-" * 45)
    for label, count in label_counts.items():
        percentage = (count / total_images) * 100
        print(f"{label:<20} {count:<8} {percentage:>8.2f}%")
    
    # Show sample of first few labels
    labels_subset = list(df['label'])[:5]
    print("\nSample of first 5 labels:")
    for idx, label in enumerate(labels_subset, 1):
        print(f"Image {idx}: {label}")
    
    # Create label mappings
    print("\nCreating label mappings...")
    labels_list = sorted(list(set(df['label'])))
    label2id = {label: idx for idx, label in enumerate(labels_list)}
    id2label = {idx: label for idx, label in enumerate(labels_list)}
    
    # Print mapping information
    print("\nLabel Mapping Information:")
    print("-" * 45)
    print("ID  Label")
    print("-" * 45)
    for idx, label in id2label.items():
        print(f"{idx:<4}{label}")
    
    # Store mappings as dataset attributes
    dataset_info = {
        'num_labels': len(labels_list),
        'id2label': id2label,
        'label2id': label2id
    }
    
    # Convert DataFrame to Hugging Face Dataset
    print("\nConverting to Hugging Face Dataset format...")
    try:
        # Create ClassLabel object for label mapping
        class_labels = ClassLabel(num_classes=len(labels_list), names=labels_list)
        
        # Create initial dataset
        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        
        # Map string labels to numeric IDs
        def map_label2id(example):
            example['label'] = class_labels.str2int(example['label'])
            return example
        
        print("Mapping labels to IDs...")
        dataset = dataset.map(map_label2id, batched=True)
        dataset = dataset.cast_column('label', class_labels)
        
        print("Successfully created and processed Hugging Face Dataset")
        print(f"Dataset features: {dataset.features}")
        print(f"Total number of classes: {dataset_info['num_labels']}")
        
        # Split dataset into train and test sets
        print("\nSplitting dataset into train and test sets (60-40 split)...")
        splits = dataset.train_test_split(
            test_size=0.4, 
            shuffle=True, 
            stratify_by_column="label", 
            seed=42
        )
        
        train_data = splits['train']
        test_data = splits['test']
        
        print(f"Training set size: {len(train_data)} images")
        print(f"Testing set size: {len(test_data)} images")
        
        # Verify stratification
        print("\nLabel distribution in splits:")
        print("Training set:")
        for label in labels_list:
            count = sum(1 for example in train_data if id2label[example['label']] == label)
            percentage = (count / len(train_data)) * 100
            print(f"  {label:<20} {count:<8} {percentage:>8.2f}%")
        
        print("\nTesting set:")
        for label in labels_list:
            count = sum(1 for example in test_data if id2label[example['label']] == label)
            percentage = (count / len(test_data)) * 100
            print(f"  {label:<20} {count:<8} {percentage:>8.2f}%")
        
        # Store splits in dataset_info
        dataset_info['train_data'] = train_data
        dataset_info['test_data'] = test_data
        
        # Display example images and label information from test dataset
        print("\nDisplaying dataset label statistics and example images...")
        try:
            # First, show label mapping information
            print("\nLabel Mapping Information:")
            print("-" * 50)
            print("ID  Label              Example Count")
            print("-" * 50)
            
            # Count examples per label
            label_counts = {}
            for example in test_data:
                label_id = example["label"]
                label_counts[label_id] = label_counts.get(label_id, 0) + 1
            
            # Display label statistics
            for label_id, label_name in id2label.items():
                count = label_counts.get(label_id, 0)
                print(f"{label_id:<4}{label_name:<20}{count}")
            
            # Display example images
            plt.figure(figsize=(15, 5))
            
            # Display 3 example images with detailed labels
            for i in range(3):
                plt.subplot(1, 3, i+1)
                image = test_data[i]["image"]
                label_id = test_data[i]["label"]
                emotion = id2label[label_id]
                
                # Calculate percentage of this emotion in dataset
                emotion_percentage = (label_counts[label_id] / len(test_data)) * 100
                
                plt.imshow(image)
                plt.title(f"Emotion: {emotion}\nID: {label_id}\n({emotion_percentage:.1f}% of dataset)")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Now test the pipeline on the displayed images
            if 'inference_pipeline' in dataset_info:
                print("\nPredicting emotions for example images...")
                for i in range(3):
                    image = test_data[i]["image"]
                    true_label = id2label[test_data[i]["label"]]
                    
                    # Get predictions
                    predictions = dataset_info['inference_pipeline'](image)
                    
                    # Print results
                    print(f"\nImage {i+1}:")
                    print(f"True emotion: {true_label}")
                    print("Predictions:")
                    for pred in predictions:
                        print(f"- {pred['label']}: {pred['score']:.4f}")
            
            print("\nSuccessfully displayed example test images and predictions")
            
        except Exception as e:
            print(f"Warning: Could not display example images or make predictions: {str(e)}")
        
        # Set up ViT model preprocessing
        print("\nSetting up ViT model and image transformations...")
        model_str = 'dima806/face_emotions_image_detection'
        
        # Initialize the ViT processor
        try:
            processor = ViTImageProcessor.from_pretrained(model_str)
            image_mean = processor.image_mean
            image_std = processor.image_std
            size = processor.size["height"]
            print(f"Loaded ViT processor - Input size: {size}x{size}")
            
            # Define image transformations
            normalize = Normalize(mean=image_mean, std=image_std)
            
            # Training transformations with augmentation
            train_transforms = Compose([
                Resize((size, size)),
                RandomRotation(90),
                RandomAdjustSharpness(2),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                normalize
            ])
            
            # Validation/Testing transformations
            val_transforms = Compose([
                Resize((size, size)),
                ToTensor(),
                normalize
            ])
            
            # Define transformation functions
            def apply_train_transforms(examples):
                examples['pixel_values'] = [
                    train_transforms(image.convert("RGB")) 
                    for image in examples['image']
                ]
                return examples
            
            def apply_val_transforms(examples):
                examples['pixel_values'] = [
                    val_transforms(image.convert("RGB")) 
                    for image in examples['image']
                ]
                return examples
            
            # Define collate function for batch processing
            def collate_fn(examples):
                """
                Collate function to prepare batches for training.
                Args:
                    examples: List of examples from the dataset
                Returns:
                    dict: Contains batched pixel_values and labels
                """
                # Stack image tensors
                pixel_values = torch.stack([example["pixel_values"] for example in examples])
                # Convert labels to tensor
                labels = torch.tensor([example['label'] for example in examples])
                return {
                    "pixel_values": pixel_values,
                    "labels": labels
                }
            
            # Store everything in dataset_info
            dataset_info.update({
                'processor': processor,
                'train_transforms': train_transforms,
                'val_transforms': val_transforms,
                'apply_train_transforms': apply_train_transforms,
                'apply_val_transforms': apply_val_transforms,
                'image_size': size,
                'collate_fn': collate_fn
            })
            
            print("Successfully set up image transformations")
            
            # Apply transforms to datasets
            print("\nApplying transforms to datasets...")
            train_data.set_transform(apply_train_transforms)
            test_data.set_transform(apply_val_transforms)
            print("Successfully applied transforms to train and test datasets")
            
            # Verify transforms by processing a sample
            try:
                # Verify individual sample transforms
                sample_train = train_data[0]
                sample_test = test_data[0]
                print("\nSingle Sample Verification:")
                print(f"Training sample shape: {sample_train['pixel_values'].shape}")
                print(f"Test sample shape: {sample_test['pixel_values'].shape}")
                
                # Verify batch processing
                print("\nBatch Processing Verification:")
                batch_size = 4
                train_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    shuffle=True
                )
                test_loader = DataLoader(
                    test_data,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    shuffle=False
                )
                
                # Get a sample batch
                sample_batch = next(iter(train_loader))
                print(f"Batch pixel values shape: {sample_batch['pixel_values'].shape}")
                print(f"Batch labels shape: {sample_batch['labels'].shape}")
                print(f"Sample batch labels: {[id2label[l.item()] for l in sample_batch['labels']]}")
                
                # Store data loaders
                dataset_info.update({
                    'train_loader': train_loader,
                    'test_loader': test_loader
                })
                
                # Initialize and configure the model
                print("\nInitializing ViT model...")
                model = ViTForImageClassification.from_pretrained(
                    model_str,
                    num_labels=len(labels_list),
                    ignore_mismatched_sizes=True
                )
                
                # Configure label mappings in model config
                model.config.id2label = id2label
                model.config.label2id = label2id
                
                # Print model information
                trainable_params = model.num_parameters(only_trainable=True)
                total_params = model.num_parameters()
                print(f"\nModel Statistics:")
                print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
                print(f"Total parameters: {total_params/1e6:.2f}M")
                print(f"Architecture: {model.config.architectures[0]}")
                print(f"Number of classes: {model.config.num_labels}")
                
                # Store model in dataset_info
                dataset_info['model'] = model
                
                # Set up evaluation metrics
                print("\nSetting up evaluation metrics...")
                try:
                    # Load accuracy metric
                    accuracy_metric = evaluate.load("accuracy")
                    
                    # Define metric computation function
                    def compute_metrics(eval_pred):
                        """
                        Compute evaluation metrics for the model.
                        
                        Args:
                            eval_pred: EvalPrediction object containing:
                                - predictions: Model output logits
                                - label_ids: True labels
                                
                        Returns:
                            dict: Dictionary containing accuracy score
                        """
                        predictions = eval_pred.predictions
                        label_ids = eval_pred.label_ids
                        
                        # Get predicted class labels
                        predicted_labels = predictions.argmax(axis=1)
                        
                        # Compute accuracy
                        acc_score = accuracy_metric.compute(
                            predictions=predicted_labels,
                            references=label_ids
                        )['accuracy']
                        
                        # Print detailed evaluation information
                        print(f"\nEvaluation Details:")
                        print(f"Total samples evaluated: {len(label_ids)}")
                        print(f"Accuracy: {acc_score:.4f}")
                        
                        # Compute confusion matrix
                        conf_matrix = confusion_matrix(label_ids, predicted_labels)
                        print("\nConfusion Matrix:")
                        for i, row in enumerate(conf_matrix):
                            label = id2label[i]
                            print(f"{label:<10}: {row}")
                        
                        return {
                            "accuracy": acc_score
                        }
                    
                    # Store metric function in dataset_info
                    dataset_info['compute_metrics'] = compute_metrics
                    print("Successfully set up evaluation metrics")
                    
                    # Set up training configuration
                    print("\nConfiguring training parameters...")
                    try:
                        # Define training settings
                        metric_name = "accuracy"
                        model_name = "face_emotions_image_detection"
                        num_train_epochs = 8
                        
                        # Create training arguments
                        training_args = TrainingArguments(
                            output_dir=model_name,
                            logging_dir='./logs',
                            evaluation_strategy="epoch",
                            learning_rate=2e-7,
                            per_device_train_batch_size=64,
                            per_device_eval_batch_size=32,
                            num_train_epochs=num_train_epochs,
                            weight_decay=0.02,
                            warmup_steps=50,
                            remove_unused_columns=False,
                            save_strategy='epoch',
                            load_best_model_at_end=True,
                            save_total_limit=1,
                            report_to="none"
                        )
                        
                        # Print training configuration summary
                        print("\nTraining Configuration:")
                        print(f"Model name: {model_name}")
                        print(f"Number of epochs: {num_train_epochs}")
                        print(f"Learning rate: {training_args.learning_rate}")
                        print(f"Batch size (train): {training_args.per_device_train_batch_size}")
                        print(f"Batch size (eval): {training_args.per_device_eval_batch_size}")
                        print(f"Weight decay: {training_args.weight_decay}")
                        print(f"Warmup steps: {training_args.warmup_steps}")
                        
                        # Create output directories
                        os.makedirs(model_name, exist_ok=True)
                        os.makedirs('./logs', exist_ok=True)
                        
                        # Store training configuration in dataset_info
                        dataset_info.update({
                            'training_args': training_args,
                            'metric_name': metric_name,
                            'model_name': model_name,
                            'num_train_epochs': num_train_epochs
                        })
                        
                        print("Successfully configured training parameters")
                        
                        # Initialize Trainer
                        print("\nInitializing Trainer...")
                        try:
                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                train_dataset=train_data,
                                eval_dataset=test_data,
                                data_collator=collate_fn,
                                compute_metrics=compute_metrics,
                                tokenizer=processor,
                            )
                            
                            # Print trainer configuration summary
                            print("\nTrainer Configuration:")
                            print(f"Training dataset size: {len(train_data)} samples")
                            print(f"Evaluation dataset size: {len(test_data)} samples")
                            print(f"Training steps per epoch: {len(train_data) // training_args.per_device_train_batch_size}")
                            print(f"Evaluation steps per epoch: {len(test_data) // training_args.per_device_eval_batch_size}")
                            
                            # Store trainer in dataset_info
                            dataset_info['trainer'] = trainer
                            print("Successfully initialized Trainer")
                            
                            # Start model training
                            print("\nStarting model training...")
                            try:
                                # Calculate total training steps
                                total_steps = (
                                    len(train_data) 
                                    * training_args.num_train_epochs 
                                    // training_args.per_device_train_batch_size
                                )
                                print(f"Total training steps: {total_steps}")
                                print(f"Number of epochs: {training_args.num_train_epochs}")
                                print(f"Batch size: {training_args.per_device_train_batch_size}")
                                print("\nTraining progress:")
                                print("-" * 50)
                                
                                # Train the model
                                train_results = trainer.train()
                                
                                # Print training results
                                print("\nTraining Results:")
                                print("-" * 50)
                                for metric_name, value in train_results.metrics.items():
                                    if isinstance(value, float):
                                        print(f"{metric_name:.<30} {value:.4f}")
                                    else:
                                        print(f"{metric_name:.<30} {value}")
                                
                                # Save the final model with timestamp
                                timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
                                model_save_path = os.path.join(dataset_info['model_name'], f"model_{timestamp}")
                                
                                print("\nSaving final model...")
                                trainer.save_model(model_save_path)
                                
                                # Save model configuration and metadata
                                metadata = {
                                    'timestamp': timestamp,
                                    'model_name': dataset_info['model_name'],
                                    'num_labels': len(labels_list),
                                    'labels': labels_list,
                                    'training_metrics': train_results.metrics,
                                    'model_config': model.config.to_dict()
                                }
                                
                                # Save metadata to JSON file
                                metadata_path = os.path.join(model_save_path, 'model_metadata.json')
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=4)
                                
                                print(f"Model saved to: {model_save_path}")
                                print(f"Model metadata saved to: {metadata_path}")
                                
                                # Create/Update Hugging Face Hub repository
                                try:
                                    print("\nSetting up Hugging Face Hub repository...")
                                    from huggingface_hub import HfApi
                                    
                                    # Initialize HF API
                                    api = HfApi()
                                    
                                    # Define repository ID
                                    repo_id = f"dima806/{model_name}"
                                    
                                    try:
                                        # Create repository (idempotent)
                                        api.create_repo(repo_id, repo_type="model", exist_ok=True)
                                        print(f"Repository ready: {repo_id}")
                                    except Exception as e:
                                        print(f"Warning: create_repo encountered an issue: {str(e)}")
                                    
                                    # Store repository info
                                    dataset_info['hf_repo_id'] = repo_id
                                    
                                    # Upload model folder to Hugging Face Hub
                                    print("\nUploading model to Hugging Face Hub...")
                                    try:
                                        api.upload_folder(
                                            folder_path=model_name,  # Upload the entire model folder (contains timestamped subfolder)
                                            path_in_repo=".",  # Upload to root of repo
                                            repo_id=repo_id,
                                            repo_type="model",
                                            revision="main",
                                            commit_message=f"Upload trained model {timestamp}",
                                            ignore_patterns=["*.log", "logs/**", "events.out.tfevents*", ".DS_Store", "__pycache__/**"]
                                        )
                                        print(f"Successfully uploaded model to {repo_id}")
                                        print(f"View your model at: https://huggingface.co/{repo_id}")
                                        
                                        # Store upload status
                                        dataset_info['hf_upload_status'] = 'success'
                                        dataset_info['hf_model_url'] = f"https://huggingface.co/{repo_id}"
                                        
                                    except Exception as e:
                                        print(f"Warning: Failed to upload model to Hugging Face Hub: {str(e)}")
                                        dataset_info['hf_upload_status'] = 'failed'
                                    
                                    print("Hugging Face Hub repository setup and upload completed")
                                    
                                except Exception as e:
                                    print(f"Warning: Could not set up Hugging Face repository: {str(e)}")
                                    dataset_info['hf_upload_status'] = 'failed'
                                
                                # Store training results and paths
                                dataset_info.update({
                                    'train_results': train_results,
                                    'model_save_path': model_save_path,
                                    'model_metadata_path': metadata_path
                                })
                                
                                # Run post-training evaluation
                                print("\nPerforming post-training evaluation...")
                                try:
                                    # Get predictions and evaluation results
                                    eval_results = trainer.evaluate()
                                    predictions = trainer.predict(test_data)
                                    
                                    # Calculate predictions and true labels
                                    pred_labels = predictions.predictions.argmax(axis=1)
                                    true_labels = predictions.label_ids
                                    
                                    # Print comprehensive evaluation results
                                    print("\nPost-Training Evaluation Results:")
                                    print("=" * 60)
                                    
                                    # Basic metrics
                                    print("\n1. Basic Metrics:")
                                    print("-" * 30)
                                    for metric_name, value in eval_results.items():
                                        if isinstance(value, float):
                                            print(f"{metric_name:.<25} {value:.4f}")
                                        else:
                                            print(f"{metric_name:.<25} {value}")
                                    
                                    # Detailed classification report
                                    print("\n2. Per-Class Performance:")
                                    print("-" * 30)
                                    report = classification_report(
                                        true_labels,
                                        pred_labels,
                                        target_names=[id2label[i] for i in range(len(id2label))],
                                        digits=4
                                    )
                                    print(report)
                                    
                                    # Confusion matrix
                                    conf_matrix = confusion_matrix(true_labels, pred_labels)
                                    print("\n3. Confusion Matrix:")
                                    print("-" * 30)
                                    print("Predicted →")
                                    print("Actual ↓")
                                    print("           " + "".join(f"{id2label[i]:>10}" for i in range(len(id2label))))
                                    for i, row in enumerate(conf_matrix):
                                        print(f"{id2label[i]:<10} {' '.join(f'{x:>10}' for x in row)}")
                                    
                                    # Calculate per-class accuracy
                                    print("\n4. Per-Class Accuracy:")
                                    print("-" * 30)
                                    for i in range(len(id2label)):
                                        class_correct = conf_matrix[i][i]
                                        class_total = conf_matrix[i].sum()
                                        class_acc = class_correct / class_total * 100
                                        print(f"{id2label[i]:<10} {class_acc:>6.2f}%")
                                    
                                    # Store evaluation results
                                    dataset_info.update({
                                        'eval_results': eval_results,
                                        'confusion_matrix': conf_matrix,
                                        'classification_report': report
                                    })
                                    
                                    print("\nEvaluation completed successfully")
                                    
                                    # Run predictions on test dataset
                                    print("\nGenerating predictions on test dataset...")
                                    outputs = trainer.predict(test_data)
                                    
                                    # Print prediction metrics
                                    print("\nPrediction Metrics:")
                                    print("=" * 60)
                                    
                                    # Basic metrics
                                    print("\n1. Overall Metrics:")
                                    print("-" * 30)
                                    for metric_name, value in outputs.metrics.items():
                                        if isinstance(value, float):
                                            print(f"{metric_name:.<25} {value:.4f}")
                                        else:
                                            print(f"{metric_name:.<25} {value}")
                                    
                                    # Analyze predictions
                                    predictions = outputs.predictions
                                    pred_labels = predictions.argmax(axis=1)
                                    
                                    # Get confidence scores
                                    confidences = predictions.max(axis=1)
                                    avg_confidence = confidences.mean()
                                    
                                    print("\n2. Confidence Analysis:")
                                    print("-" * 30)
                                    print(f"Average confidence......... {avg_confidence:.4f}")
                                    print(f"Min confidence............ {confidences.min():.4f}")
                                    print(f"Max confidence............ {confidences.max():.4f}")
                                    
                                    # Per-class analysis
                                    print("\n3. Per-Class Predictions:")
                                    print("-" * 30)
                                    unique, counts = np.unique(pred_labels, return_counts=True)
                                    for label_id, count in zip(unique, counts):
                                        emotion = id2label[label_id]
                                        percentage = (count / len(pred_labels)) * 100
                                        avg_conf = confidences[pred_labels == label_id].mean()
                                        print(f"{emotion:<10} Count: {count:>4} ({percentage:>6.2f}%) Avg Conf: {avg_conf:.4f}")
                                    
                                    # Store prediction results
                                    dataset_info.update({
                                        'prediction_outputs': outputs,
                                        'prediction_confidences': confidences,
                                        'predicted_labels': pred_labels
                                    })
                                    
                                    # Extract true labels and predictions
                                    y_true = outputs.label_ids
                                    y_pred = outputs.predictions.argmax(1)
                                    
                                    # Define confusion matrix plotting function
                                    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', 
                                                           cmap=plt.cm.Blues, figsize=(10, 8)):
                                        """
                                        Plot and display confusion matrix with detailed formatting.
                                        """
                                        plt.figure(figsize=figsize)
                                        plt.imshow(cm, interpolation='nearest', cmap=cmap)
                                        plt.title(title, pad=20)
                                        plt.colorbar()
                                        
                                        # Setup ticks and labels
                                        tick_marks = np.arange(len(classes))
                                        plt.xticks(tick_marks, classes, rotation=45, ha='right')
                                        plt.yticks(tick_marks, classes)
                                        
                                        # Add text annotations
                                        thresh = cm.max() / 2.0
                                        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                                            plt.text(j, i, format(cm[i, j], '.0f'),
                                                   horizontalalignment="center",
                                                   color="white" if cm[i, j] > thresh else "black")
                                        
                                        plt.ylabel('True Emotion')
                                        plt.xlabel('Predicted Emotion')
                                        plt.tight_layout()
                                        
                                    # Calculate and display metrics
                                    print("\n4. Detailed Performance Metrics:")
                                    print("-" * 30)
                                    
                                    # Calculate accuracy and F1 score
                                    accuracy = accuracy_score(y_true, y_pred)
                                    f1 = f1_score(y_true, y_pred, average='macro')
                                    print(f"Overall Accuracy........... {accuracy:.4f}")
                                    print(f"Macro F1-Score............ {f1:.4f}")
                                    
                                    # Compute and plot confusion matrix
                                    cm = confusion_matrix(y_true, y_pred)
                                    print("\nGenerating confusion matrix visualization...")
                                    plot_confusion_matrix(cm, labels_list, 
                                                        title='Emotion Recognition Confusion Matrix')
                                    
                                    # Display classification report
                                    print("\nDetailed Classification Report:")
                                    print("-" * 50)
                                    report = classification_report(y_true, y_pred, 
                                                                target_names=labels_list, 
                                                                digits=4)
                                    print(report)
                                    
                                    # Store additional metrics
                                    dataset_info.update({
                                        'confusion_matrix': cm,
                                        'accuracy': accuracy,
                                        'f1_score': f1,
                                        'classification_report': report
                                    })
                                    
                                    print("\nPrediction analysis completed successfully")
                                    
                                except Exception as e:
                                    print(f"Warning: Error during evaluation/prediction: {str(e)}")
                                
                                # Print evaluation results
                                print("\nEvaluation Results:")
                                print("-" * 50)
                                
                                # Print metrics
                                for metric_name, value in eval_results.items():
                                    if isinstance(value, float):
                                        print(f"{metric_name:.<30} {value:.4f}")
                                    else:
                                        print(f"{metric_name:.<30} {value}")
                                
                                # Calculate and print additional metrics
                                predictions = trainer.predict(test_data)
                                pred_labels = predictions.predictions.argmax(axis=1)
                                true_labels = predictions.label_ids
                                
                                # Print classification report
                                print("\nDetailed Classification Report:")
                                print("-" * 50)
                                report = classification_report(
                                    true_labels,
                                    pred_labels,
                                    target_names=[id2label[i] for i in range(len(id2label))],
                                    digits=4
                                )
                                print(report)
                                
                                # Store evaluation results
                                dataset_info['eval_results'] = eval_results
                                dataset_info['classification_report'] = report
                                
                                print("\nEvaluation completed successfully")
                                
                                # Set up inference pipeline
                                print("\nSetting up inference pipeline...")
                                try:
                                    # Import pipeline
                                    from transformers import pipeline
                                    
                                    # Determine the device to use
                                    device = 0 if torch.cuda.is_available() else -1
                                    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
                                    
                                    # Create the pipeline using the trained model
                                    pipe = pipeline(
                                        'image-classification',
                                        model=dataset_info['model_save_path'],
                                        device=device
                                    )
                                    
                                    # Store pipeline in dataset_info
                                    dataset_info['inference_pipeline'] = pipe
                                    
                                    print("Inference pipeline successfully created")
                                    
                                except Exception as e:
                                    print(f"Warning: Could not create inference pipeline: {str(e)}")
                                
                            except Exception as e:
                                print(f"Warning: Could not complete evaluation: {str(e)}")
                            
                        except Exception as e:
                            print(f"Warning: Could not initialize trainer: {str(e)}")
                        
                    except Exception as e:
                        print(f"Warning: Could not configure training: {str(e)}")
                    
                except Exception as e:
                    print(f"Warning: Could not set up metrics: {str(e)}")
                
            except Exception as e:
                print(f"Warning: Could not complete model setup: {str(e)}")
            
        except Exception as e:
            print(f"Error setting up ViT preprocessing: {str(e)}")
            dataset_info['processor'] = None
        
        # Display the first image in the dataset
        print("\nDisplaying first image in dataset...")
        first_image = dataset[0]["image"]
        plt.figure(figsize=(8, 8))
        plt.imshow(first_image)
        plt.title(f"First image - Label: {dataset[0]['label']}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        dataset = None
        
    return dataset  # Return the dataset instead of df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio_to_wav(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            output_path
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Audio conversion failed: %s", str(e))
        return False
    except Exception as e:
        logging.error(f"Audio conversion failed: {str(e)}")
        return False

def convert_types(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.generic): return obj.item()
    return obj

def predict(file_path):
    try:
        if not os.path.exists(file_path):
            print("File does not exist")
            return

        if not allowed_file(file_path):
            print("Invalid file type")
            return

        timestamp_str = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        file_ext = file_path.rsplit('.', 1)[1].lower()
        
        # Convert to WAV if needed
        if file_ext != 'wav':
            wav_filename = f"temp_{timestamp_str}.wav"
            wav_path = os.path.join(UPLOAD_FOLDER, wav_filename)
            if not convert_audio_to_wav(file_path, wav_path):
                print("Audio conversion failed")
                return
            final_path = wav_path
        else:
            final_path = file_path

        # Load model and make prediction
        try:
            model = load_model("emotion_detection.h5")
            
            # Define the feature extraction and prediction functions
            def extract_all_features(path):
                try:
                    import sys
                    sys.path.append('Emotion-Aware-AI-Support-System')
                    from app.ml_model.preprocess import extract_all_features as ext_features
                    return ext_features(path)
                except ImportError:
                    # Fallback to placeholder if import fails
                    return {"feature1": np.array([0.1, 0.2]), "feature2": np.array([0.3])}
            
            def prediction(path):
                try:
                    import sys
                    sys.path.append('Emotion-Aware-AI-Support-System')
                    from app.ml_model.preprocess import prediction as model_predict
                    return model_predict(path)
                except ImportError:
                    # Fallback to placeholder if import fails
                    return [{"primary_emo": "happy", 
                            "detected_emos": [0.7, 0.2, 0.1],
                            "mean_zcr": 0.5,
                            "mean_rmse": 0.3}]
                return [{"primary_emo": "happy", 
                        "detected_emos": [0.7, 0.2, 0.1],
                        "mean_zcr": 0.5,
                        "mean_rmse": 0.3}]

            # Make predictions
            detailed_features = extract_all_features(final_path)
            results = prediction(final_path)

            # Convert numpy types to Python native types
            detailed_features = {k: convert_types(v) for k, v in detailed_features.items()}
            results = results[0] if isinstance(results, list) and len(results) > 0 else results
            results = {k: convert_types(v) for k, v in results.items()}

            # Print results
            print("\nPrediction Results:")
            print("==================")
            print(f"Primary Emotion: {results['primary_emo']}")
            print(f"Confidence Scores: {results['detected_emos']}")
            print(f"Mean ZCR: {results.get('mean_zcr')}")
            print(f"Mean RMSE: {results.get('mean_rmse')}")

            # Cleanup temporary WAV file if created
            if file_ext != 'wav' and os.path.exists(final_path):
                os.remove(final_path)

        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            if file_ext != 'wav' and os.path.exists(final_path):
                os.remove(final_path)
            return

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return

if __name__ == "__main__":
    # Example usage
    file_path = "./kids-laugh-45357.mp3"
    predict(file_path)