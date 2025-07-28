from typing import Any, Dict, Union
import numpy as np
from datasets import Dataset, load_dataset, Audio


def _validate_dataset_split(dataset_split, split_name: str):
    """Validate that a dataset split is loaded correctly."""
    if dataset_split is None:
        raise ValueError(f"{split_name} split is None")
    
    if len(dataset_split) == 0:
        raise ValueError(f"{split_name} split is empty")
    
    required_columns = {"audio", "labels"}
    missing_columns = required_columns - set(dataset_split.column_names)
    if missing_columns:
        raise ValueError(f"{split_name} split missing required columns: {missing_columns}. Available columns: {dataset_split.column_names}. Maybe renaming is enough?")
    
    try:
        first_sample = dataset_split[0]
        
        if "audio" not in first_sample or first_sample["audio"] is None:
            raise ValueError(f"{split_name} split has invalid audio data")
        
        if "labels" not in first_sample or not isinstance(first_sample["labels"], str):
            raise ValueError(f"{split_name} split has invalid labels data")
            
    except Exception as e:
        raise ValueError(f"Error validating {split_name} split: {str(e)}")


def _load_librispeech_dataset_clean():
    train = load_dataset("openslr/librispeech_asr", split="train.clean.100")
    
    # ---- load & preprocess train ----
    train = train.rename_column("text", "labels")
    train = train.map(lambda x: {"labels": x["labels"].lower()})
    train = train.cast_column("audio", Audio(sampling_rate=16_000))

    # ---- load & preprocess val ----
    validation = load_dataset("openslr/librispeech_asr", split="validation.clean")
    validation = validation.rename_column("text", "labels")
    validation = validation.map(lambda x: {"labels": x["labels"].lower()})
    validation = validation.cast_column("audio", Audio(sampling_rate=16_000))

    def __get_audio_length(example):
        return {"audio_length": len(example["audio"]["array"]) / example["audio"]["sampling_rate"]}

    validation = validation.map(__get_audio_length)  # only validation has audio > 30 seconds
    validation = validation.filter(lambda example: example["audio_length"] <= 30)

    _validate_dataset_split(train, "train")
    _validate_dataset_split(validation, "validation")

    return train, validation


def _load_multimed_german():
    dataset = load_dataset("leduckhai/MultiMed", "German")
    
    
    # ---- load & preprocess train ----
    train = dataset["train"]
    train = train.rename_column("text", "labels")
    train = train.cast_column("audio", Audio(sampling_rate=16_000))
    train = train.filter(lambda example: example["duration"] <= 30)
    
    # ---- load & preprocess val ----
    validation = dataset["eval"]
    validation = validation.rename_column("text", "labels")
    validation = validation.cast_column("audio", Audio(sampling_rate=16_000))
    validation = validation.filter(lambda example: example["duration"] <= 30)

    _validate_dataset_split(train, "train")
    _validate_dataset_split(validation, "validation")

    return train, validation


def load_train_val_data(dataset_cfg: Any):
    dataset_name = dataset_cfg.get("name")
    if dataset_name not in ["librispeech_clean", "multimed_german"]:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from 'librispeech_clean' or 'multimed_german'.")
    
    if dataset_name == "librispeech_clean":
        print("Loading LibriSpeech clean dataset (~120 GB)...")
        return _load_librispeech_dataset_clean()
    elif dataset_name == "multimed_german":
        print("Loading MultiMed German dataset (~5 GB)...")
        return _load_multimed_german()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    