# Voxtral Transcription Fine-Tuning [Experimental]

This repository contains a fine-tuning script for the transcription task of Mistral's [Voxtral](https://huggingface.co/docs/transformers/main/model_doc/voxtral) model using the Huggingface `transformers` Trainer. It also includes several utilities that can be useful for training Automatic Speech Recognition (ASR) systems: Length-Aware Audiomentations + WER calculation.


### ⚠️ Important Disclaimer

This is **research code** and has not been extensively tested. It is primarily intended as a starting point for those experimenting with Voxtral or implementing custom fine-tuning functionalities, such as [Prompt Biasing](https://arxiv.org/abs/2506.06252). This is full-finetune code for the 3B Voxtral model. It requires ~80 GB VRAM to run with batch size 8 on bf16 w/ adamw_bnb_8bit optimizer, or ~47 GB VRAM for batch size 4. The entire pipeline has only been used with audio clips under 30 seconds. Performance and stability with longer audio have not been verified. For a more stable and optimized implementation, please refer to the work being done by **Unsloth**. They have announced upcoming support for Voxtral fine-tuning. You can track their progress here: [Unsloth GitHub Issue #3013](https://github.com/unslothai/unsloth/issues/3013). 


## Getting Started

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) to enjoy this project. 
### 1. Requirements

To setup and run this project simply:
```commandline
uv sync
```
Activate the environment:
```commandline
source .venv/bin/activate
```

### 2. Dataset Setup
#### Examples
We provide implementations for the [MultiMed](https://huggingface.co/leduckhai/MultiMed) dataet (german only) as example. If you run the script (see 3. Run Training), the dataset downloads automatically (~5 GB). It is recommended to start with this smaller dataset and watch the eval loss go down.

We also provide [Librispeech](https://huggingface.co/datasets/openslr/librispeech_asr) as example. If you want to train on (a subset of) Librispeech, change the dataset name and language in the config file. Warning: this will automatically download the dataset from Huggingface to your disk (~120 GB).

#### Custom

If you like to implement a custom dataset, please use the example dataset as reference (`voxtral_finetune/datasets.py`) and specify in the config file. Each sample in the dataset must contain:
-   **audio**: A `datasets.Audio` object (a dictionary with `array` and `sampling_rate`).
-   **label**: A string containing the ground-truth transcription.

The current implementation assumes that the audio sampling rate is constant within each batch. This is standard behavior for `datasets`, but please verify and preprocess if required. Please resample the dataset to 16000 Hz.

Please also change the language in the dataset settings, e.g. from "en" to "de" if you do german ASR. Mixed language training is currently not supported.


### 3. Run Training

Single GPU:
```bash
uv run voxtral-finetune  # to run w/ default config: 'voxtral-3B-v0'
```
```bash
uv run voxtral-finetune your_config_name  # point at a different config file
```

Multi-GPU:
```bash
accelerate launch ./src/voxtral_finetune/main.py your_config_name
```

---

## Core Components

### `VoxtralCollatorTranscriptionTask`

This custom data collator prepares a batch for training. It takes the audio and text from the dataset and correctly formats them into the prompt-completion structure required by the transcription task.

**How it works:**

1.  Generate Transcription Prompt: It first creates the instruction part of the sequence including the audio placeholders, e.g.:
    ```
    <s> [INST] [BEGIN_AUDIO] [AUDIO] ... [AUDIO] [/INST] lang:de [TRANSCRIBE]
    ```

2.  Generate the Completion: It then tokenizes the target transcription, which will serve as the completion part of the sequence.
    ```
    Hallo Welt </s>
    ```

3.  **Concatenate and Mask**: The collator concatenates the prompt and completion token IDs. It then applies **left-padding** to the combined sequence and creates the attention mask. Crucially, it sets the labels for the prompt tokens to `-100` so that the loss is only calculated on the completion (the actual transcription).

### `CustomSeq2SeqTrainer`

This custom `Trainer` inherits from the standard `Seq2SeqTrainer` but modifies the prediction step to work correctly with Voxtral's prompt-based generation.

**Why is it needed?**

During training, we feed the model the entire sequence (prompt + completion) and just mask the loss on the prompt. However, during inference (evaluation/generation), we must only provide the **prompt** and let the model generate the completion.

The `CustomSeq2SeqTrainer` implements a custom `prediction_step` that:
1.  Takes the full input sequence from the evaluation dataloader.
2.  Slices it to remove the ground-truth completion part, leaving only the prompt.
3.  Feeds this prompt to the `model.generate()` function to produce a prediction.
4.  Handles the necessary padding adjustments for the generation process.

### `build_wer_fn`

This is a helper function for Word Error Rate (WER) calculation. A key challenge with Voxtral is that the language identifier (e.g., `lang:de`) is part of the generated text and is not a special token, therefore not skipped when generating. Standard decoding would leave it in the final output, corrupting the WER score. Therefore, this function post-processes the generated text by using a regular expression to find and strip away the `lang:xx` token before passing the clean prediction and reference to the `jiwer` library.

---


## Future improvements

- Mixed language training by passing the correct language token to the collator dynamically
- Instruction task using the chat template

