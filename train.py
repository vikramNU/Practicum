import pandas as pd
import torch
import pandas as pd
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer

df=pd.read_csv('text_keywords.csv', index_col=0)

class DataModule(Dataset):
    """
    Data Module for pytorch
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        self.data = data
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        keywords_encoding = self.tokenizer(
            data_row["keywords"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_encoding = self.tokenizer(
            data_row["text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = text_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            keywords=data_row["keywords"],
            text=data_row["text"],
            keywords_input_ids=keywords_encoding["input_ids"].flatten(),
            keywords_attention_mask=keywords_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=text_encoding["attention_mask"].flatten(),
        )


class PLDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 4,
        split: float = 0.1,
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.split = split
        self.batch_size = batch_size
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_dataset = DataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = DataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """training dataloader"""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def test_dataloader(self):
        """test dataloader"""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def val_dataloader(self):
        """validation dataloader"""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


class LightningModel(pl.LightningModule):
    """PyTorch Lightning Model class"""

    def __init__(self, tokenizer, model, output: str = "outputs"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output = output

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """forward step"""
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """training step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_size):
        """validation step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_size):
        """test step"""
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        return AdamW(self.parameters(), lr=0.0001)


class trainer:
    """
    Keytotext model trainer
    """

    def __init__(self):
        pass

    def from_pretrained(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True
        )

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 4,
        max_epochs: int = 5,
        use_gpu: bool = True,
        outputdir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        test_split=0.1,
        tpu_cores = None,
    ):
        self.target_max_token_len = target_max_token_len
        self.max_epoch = max_epochs
        self.train_df = train_df
        self.test_df = test_df

        self.data_module = PLDataModule(
            train_df=train_df,
            test_df=test_df,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            split=test_split,
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer, model=self.model, output=outputdir
        )

        early_stop_callback = (
            [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
            ]
            if early_stopping_patience_epochs > 0
            else None
        )

        gpus = -1 if use_gpu else 0

        trainer = Trainer(
            callbacks=early_stop_callback,
            max_epochs=max_epochs,
            gpus=gpus,
            progress_bar_refresh_rate=5,
            tpu_cores=tpu_cores
        )

        trainer.fit(self.T5Model, self.data_module)

    def load_model(self, model_dir: str = "outputs", use_gpu: bool = False):
        self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception(
                    "exception ---> no gpu found. set use_gpu=False, to use CPU"
                )
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def save_model(self, model_dir="model"):
        path = f"{model_dir}"
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

model = trainer()
# model.from_pretrained(model_name="t5-large")

# test_df=df[800:]

# model.train(train_df=df[:800], test_df=test_df, batch_size=4, max_epochs=1, use_gpu=True)

# model.save_model('/home/ubuntu/Prac/Practicum')
models = T5ForConditionalGeneration.from_pretrained("C:\\Users\\Vikram_PC\\Downloads\\downmod")
tokenizer = T5Tokenizer.from_pretrained(f"C:\\Users\\Vikram_PC\\Downloads\\downmod")
def predict(
      keywords: list,
      max_length: int = 512,
      num_return_sequences: int = 1,
      num_beams: int = 2,
      top_k: int = 50,
      top_p: float = 0.95,
      do_sample: bool = True,
      repetition_penalty: float = 2.5,
      length_penalty: float = 1.0,
      early_stopping: bool = True,
      skip_special_tokens: bool = True,
      clean_up_tokenization_spaces: bool = True,
      use_gpu: bool = True,
  ):
      """
      generates prediction for K2T model
      Args:
          Keywords (list): any keywords for generating predictions
          max_length (int, optional): max token length of prediction. Defaults to 512.
          num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
          num_beams (int, optional): number of beams. Defaults to 2.
          top_k (int, optional): Defaults to 50.
          top_p (float, optional): Defaults to 0.95.
          do_sample (bool, optional): Defaults to True.
          repetition_penalty (float, optional): Defaults to 2.5.
          length_penalty (float, optional): Defaults to 1.0.
          early_stopping (bool, optional): Defaults to True.
          skip_special_tokens (bool, optional): Defaults to True.
          clean_up_tokenization_spaces (bool, optional): Defaults to True.
          use_gpu: Defaults to True.
      Returns:
          str: returns predictions
      """
      if use_gpu:
          if torch.cuda.is_available():
              device = torch.device("cuda")
          else:
              raise Exception(
                  "exception ---> no gpu found. set use_gpu=False, to use CPU"
              )
      else:
          device = torch.device("cpu")

      source_text = " ".join(map(str, keywords))

      input_ids = tokenizer.encode(
          source_text, return_tensors="pt", add_special_tokens=True
      )

      input_ids = input_ids
      generated_ids = models.generate(
          input_ids=input_ids,
          num_beams=num_beams,
          max_length=max_length,
          repetition_penalty=repetition_penalty,
          length_penalty=length_penalty,
          early_stopping=early_stopping,
          top_p=top_p,
          top_k=top_k,
          num_return_sequences=num_return_sequences,
      )
      preds = [
          tokenizer.decode(
              g,
              skip_special_tokens=skip_special_tokens,
              clean_up_tokenization_spaces=clean_up_tokenization_spaces,
          )
          for g in generated_ids
      ]
      
      return preds[0]
def evaluate(test_df: pd.DataFrame, metrics: str = "rouge"):
    from tqdm import tqdm
    """

    :param test_df:
    :param metrics:
    :return: Output metrics for keytotext
    
    
    """
    from datasets import load_metric
    # metric = load_metric(metrics)
    metric = load_metric("bleu")
    print(test_df)
    input_text = test_df["text"]
    references = test_df["keywords"]
    print(len(input_text))
    predictions = [predict(x) for x in tqdm(input_text)]
    predictions = []
    for x in tqdm(input_text):
      p=predict(x)
      predictions.append(p)
      print(p,x)

    results = metric.compute(predictions=predictions, references=references)
    output = results
    # output = {
    #     "Rouge 1": {
    #         "Rouge_1 Low Precision": results["rouge1"].low.precision,
    #         "Rouge_1 Low recall": results["rouge1"].low.recall,
    #         "Rouge_1 Low F1": results["rouge1"].low.fmeasure,
    #         "Rouge_1 Mid Precision": results["rouge1"].mid.precision,
    #         "Rouge_1 Mid recall": results["rouge1"].mid.recall,
    #         "Rouge_1 Mid F1": results["rouge1"].mid.fmeasure,
    #         "Rouge_1 High Precision": results["rouge1"].high.precision,
    #         "Rouge_1 High recall": results["rouge1"].high.recall,
    #         "Rouge_1 High F1": results["rouge1"].high.fmeasure,
    #     },
    #     "Rouge 2": {
    #         "Rouge_2 Low Precision": results["rouge2"].low.precision,
    #         "Rouge_2 Low recall": results["rouge2"].low.recall,
    #         "Rouge_2 Low F1": results["rouge2"].low.fmeasure,
    #         "Rouge_2 Mid Precision": results["rouge2"].mid.precision,
    #         "Rouge_2 Mid recall": results["rouge2"].mid.recall,
    #         "Rouge_2 Mid F1": results["rouge2"].mid.fmeasure,
    #         "Rouge_2 High Precision": results["rouge2"].high.precision,
    #         "Rouge_2 High recall": results["rouge2"].high.recall,
    #         "Rouge_2 High F1": results["rouge2"].high.fmeasure,
    #     },
    #     "Rouge L": {
    #         "Rouge_L Low Precision": results["rougeL"].low.precision,
    #         "Rouge_L Low recall": results["rougeL"].low.recall,
    #         "Rouge_L Low F1": results["rougeL"].low.fmeasure,
    #         "Rouge_L Mid Precision": results["rougeL"].mid.precision,
    #         "Rouge_L Mid recall": results["rougeL"].mid.recall,
    #         "Rouge_L Mid F1": results["rougeL"].mid.fmeasure,
    #         "Rouge_L High Precision": results["rougeL"].high.precision,
    #         "Rouge_L High recall": results["rougeL"].high.recall,
    #         "Rouge_L High F1": results["rougeL"].high.fmeasure,
    #     },
    #     "rougeLsum": {
    #         "rougeLsum Low Precision": results["rougeLsum"].low.precision,
    #         "rougeLsum Low recall": results["rougeLsum"].low.recall,
    #         "rougeLsum Low F1": results["rougeLsum"].low.fmeasure,
    #         "rougeLsum Mid Precision": results["rougeLsum"].mid.precision,
    #         "rougeLsum Mid recall": results["rougeLsum"].mid.recall,
    #         "rougeLsum Mid F1": results["rougeLsum"].mid.fmeasure,
    #         "rougeLsum High Precision": results["rougeLsum"].high.precision,
    #         "rougeLsum High recall": results["rougeLsum"].high.recall,
    #         "rougeLsum High F1": results["rougeLsum"].high.fmeasure,
    #     },
    # }
    return output
evaluate(test_df=df[800:])