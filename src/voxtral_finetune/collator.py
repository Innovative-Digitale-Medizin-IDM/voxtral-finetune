import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature


class VoxtralCollatorTranscriptionTask:
    def __init__(self, processor: AutoProcessor, model_id: str, language: str = "en", use_fp16: bool = False):
        self.processor = processor
        self.sep_id = 4  # ID for the [/INST] token, which denotes the beginning of the transcription.
        self.use_fp16 = use_fp16
        self.model_id = model_id
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        self.language = language


    def __call__(self, batch: list[dict]) -> dict:
        """
            Tokenization and Padding of a Huggingface dataset batch using \
                chat templates and transcription request by HF Transformers / Mistral Commons.
        """
        audio_arrays = [item["audio"]["array"] for item in batch]
        labels_text = [item["labels"] for item in batch]
        sampling_rate = batch[0]["audio"]["sampling_rate"] # assumes sampling rate is constant within batch (should be by now, otherwise please adapt dataset)
        batch_size = len(batch)

        tokenized_transcription_request = self.processor.apply_transcription_request(
            language=self.language,
            audio=audio_arrays,
            sampling_rate=sampling_rate,
            format=["wav"] * batch_size,
            model_id=self.model_id
        )
        transcription_ids_list = tokenized_transcription_request["input_ids"]

        conversations = [
            [
                {"role": "user", "content": "Some mock text"}, # The audio placeholder is implicit here
                {"role": "assistant", "content": text},
            ]
            for text in labels_text
        ]
        
        tokenized_chat_template = self.processor.apply_chat_template(
            conversations, 
            continue_final_message=True,
            return_tensors="pt",
        )
        chat_ids_tensor = tokenized_chat_template["input_ids"]

        # find seperator token, which marks the beginning of the completion (i.e. transcription)
        sep_token_pos = (chat_ids_tensor == self.sep_id).nonzero(as_tuple=True)[1]

        # cut away prompt (i.e. instruction)
        combined_ids_list = [
            torch.cat([
                transcription_ids, 
                chat_ids_tensor[i, sep_token_pos[i] + 1:]
            ])
            for i, transcription_ids in enumerate(transcription_ids_list)
        ]

        # left padding
        self.processor.tokenizer.padding_side = "left"
        padded_batch = self.processor.tokenizer.pad(
            {"input_ids": combined_ids_list},
            padding="longest",
            return_tensors="pt"
        )
        padded_input_ids = padded_batch["input_ids"]
        
        # set labels to -100 (ignored by loss) up until the start of completion
        start_of_completion_token_id = 34
        mask = (padded_input_ids == start_of_completion_token_id).long().cumsum(dim=1).sub(
            (padded_input_ids == start_of_completion_token_id).long()
        )
        labels = padded_input_ids.detach().clone()
        labels[mask == 0] = -100

        # Attention mask
        attention_mask = (padded_input_ids != self.processor.tokenizer.pad_token_id).long()

        # The input_features are the mel transformed audio
        input_features = tokenized_transcription_request["input_features"]

        if self.use_fp16:
            input_features = input_features.to(dtype=torch.float16)
   
        data =  {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "labels": labels.to(torch.long),
        }
        return BatchFeature(data=data)
