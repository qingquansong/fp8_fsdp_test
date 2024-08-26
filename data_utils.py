from abc import ABC, abstractmethod

import datasets
import lightning.pytorch as pl
from collator import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader

_RETAIN_COLUMNS = {"input_ids", "attention_mask", "labels"}
MMLU_QUESTION = "<Question>"
MMLU_CHOICES = "<Choices>"
MMLU_ANSWER = "<Answer>"
CNN_RESPONSE_TEMPLATE = "### Highlights:"


class DataModule(pl.LightningDataModule, ABC):
    def __init__(self, tokenizer, data_path, max_length, batch_size, n_train, n_val, exp_len=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.exp_len = exp_len

    @abstractmethod
    def formatting_func(self, example):
        pass

    def aug_list(self, input, exp_len=4096):
        ori_len = len(input[0])
        return [input[0] * (exp_len // ori_len) + input[0][: exp_len % ori_len]]

    def tokenize(self, example):
        outputs = self.tokenizer(
            self.formatting_func(example),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        return {
            "input_ids": self.aug_list(outputs["input_ids"], exp_len=self.exp_len),
            "attention_mask": self.aug_list(outputs["attention_mask"]),
        }

    @abstractmethod
    def setup(self, stage) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=31,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=31,
        )


class CNNModule(DataModule):
    def __init__(self, tokenizer, data_path, max_length, batch_size, n_train, n_val):
        super().__init__(tokenizer, data_path, max_length, batch_size, n_train, n_val)
        response_prompt = tokenizer.encode(
            CNN_RESPONSE_TEMPLATE, add_special_tokens=False
        )
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_prompt,
            pad_to_multiple_of=16,
        )

    def formatting_func(self, example):
        output = "Given a text, please give highlights.\n\n"
        output += f"TEXT: {example['article']}\n"
        output += f" {CNN_RESPONSE_TEMPLATE} "
        output += f"{example['highlights']} "
        return [output]

    def setup(self, stage) -> None:
        dataset = datasets.load_dataset(path=self.data_path)
        dataset["train"] = dataset["train"].select(range(self.n_train))
        dataset["test"] = dataset["test"].select(range(self.n_val))
        self.train_dataset = dataset["train"].map(
            self.tokenize,
            remove_columns=list(set(dataset["train"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.val_dataset = dataset["test"].map(
            self.tokenize,
            remove_columns=list(set(dataset["test"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )


class MMLUModule(DataModule):
    def __init__(self, tokenizer, data_path, max_length, batch_size, n_train, n_val, exp_len=4096):
        super().__init__(tokenizer, data_path, max_length, batch_size, n_train, n_val, exp_len)
        response_prompt = tokenizer.encode(f"{MMLU_ANSWER}", add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_prompt,
            pad_to_multiple_of=16,
        )

    def formatting_func(self, example):
        output_texts = []
        for i in range(len(example["question"])):
            choices = ""
            for j in range(len(example["choices"][i])):
                choices += f"{j+1}. {example['choices'][i][j]}; "
            s = "Below is a question and multiple choice answers, choices separated by a semicolon. Please select the best answer for the question. "
            s += f"{MMLU_QUESTION}{example['question'][i]} "
            s += f"{MMLU_CHOICES}{choices} "
            s += f" {MMLU_ANSWER} {example['answer'][i]}"
            output_texts.append(s)
        return output_texts

    def setup(self, stage) -> None:
        dataset = datasets.load_from_disk(self.data_path)["auxiliary_train"]
        dataset = dataset.train_test_split(test_size=self.n_val, seed=42)
        dataset["train"] = dataset["train"].select(range(self.n_train))
        self.train_dataset = dataset["train"].map(
            self.tokenize,
            remove_columns=list(set(dataset["train"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
        )
        self.val_dataset = dataset["test"].map(
            self.tokenize,
            remove_columns=list(set(dataset["test"].column_names) - _RETAIN_COLUMNS),
            batched=True,
            batch_size=1,
            )

