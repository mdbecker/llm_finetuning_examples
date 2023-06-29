## Mixed-Precision Training in Text Summarization

The provided script demonstrates how to perform text summarization with hyperparameter optimization and mixed-precision training. While the code is self-explanatory, let's highlight some of the key benefits and considerations of using mixed-precision training.

### Integration of Accelerate Library

The script utilizes the [Accelerate library](https://github.com/huggingface/accelerate) which is designed to facilitate training processes. This library is particularly useful as it allows the same PyTorch code to be compatible with different hardware configurations (such as multi-GPU setups) without extensive customizations.

### Benefits of Mixed-Precision Training

Mixed-precision training uses a combination of single-precision (32-bit) and half-precision (16-bit) floating-point numbers during training. It has several benefits:

#### Memory Savings

When working with large models, especially deep learning models, GPU memory can be a constraint. Mixed-precision training significantly reduces the memory footprint, allowing for larger batch sizes or enabling the training of models that would not otherwise fit in memory.

#### Performance Improvements

Training on GPUs that support Tensor Cores can benefit from hardware acceleration for specific operations, leading to faster training times with mixed precision.

### Cautions for Mixed-Precision Training

Although mixed-precision training has its benefits, it also comes with challenges:

#### Numerical Instability

Reduced precision can sometimes lead to numerical instability. Careful handling of loss scaling and learning rate schedules may be necessary to mitigate this.

### Diff
The difference between the original code [base.md](base.md) and this new version are:

```diff
@@ -2,6 +2,7 @@ import numpy as np
 import optuna
 from datasets import load_dataset, load_metric
 from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
+from accelerate import Accelerator  # ADDED: Import the Accelerate library to enable mixed-precision training
 
 # Data and model loading
 raw_datasets = load_dataset("xsum")
@@ -36,14 +37,22 @@ def optuna_hp_space(trial):
         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
     }
 
+# Accelerator
+accelerator = Accelerator(mixed_precision=True)  # ADDED: Initialize the Accelerator with mixed-precision enabled. This will allow the model to use less memory and potentially speed up training by using half-precision floating points.
+
 # Model initialization function for Optuna
 def model_init(trial):
     model = LlamaForCausalLM.from_pretrained(model_checkpoint)
     # handle pad_token issue
     if padding_added:
         model.resize_token_embeddings(len(tokenizer))
+    # Prepare model with accelerator
+    model = accelerator.prepare(model)  # ADDED: Prepare the model using the accelerator's prepare function. This step ensures that the model is compatible with mixed-precision training and distributed training if applicable.
     return model
 
+# Prepare data_collator with accelerator
+data_collator = accelerator.prepare(data_collator)  # ADDED: Also prepare the data_collator using the accelerator. This step ensures that the data collator is compatible with mixed-precision and distributed training, which is important for efficient training on modern hardware.
+
 # Training arguments
 training_args = Seq2SeqTrainingArguments(
     output_dir="llama7b-finetuned-xsum", evaluation_strategy="epoch",
```

### Full file:
```python
import numpy as np
import optuna
from datasets import load_dataset, load_metric
from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from accelerate import Accelerator  # ADDED: Import the Accelerate library to enable mixed-precision training

# Data and model loading
raw_datasets = load_dataset("xsum")
model_checkpoint = 'chaoyi-wu/PMC_LLAMA_7B'
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint)
padding_added = False
# Add padding token if it doesn't exist and resize token embeddings... this fixes an issue with loading older versions of LLAMA models using newer version of the transformers library
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    padding_added = True

# Preprocessing
def preprocess_function(examples, max_input_length=1024, max_target_length=128):
    inputs = tokenizer(examples["document"], max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt")
    labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding='max_length', return_tensors="pt")["input_ids"]
    return {**inputs, "labels": labels}

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=10)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# Metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True)
    return {k: round(v, 4) for k, v in load_metric("rouge").compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True).items()}

# Hyperparameter space for Optuna
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
    }

# Accelerator
accelerator = Accelerator(mixed_precision=True)  # ADDED: Initialize the Accelerator with mixed-precision enabled. This will allow the model to use less memory and potentially speed up training by using half-precision floating points.

# Model initialization function for Optuna
def model_init(trial):
    model = LlamaForCausalLM.from_pretrained(model_checkpoint)
    # handle pad_token issue
    if padding_added:
        model.resize_token_embeddings(len(tokenizer))
    # Prepare model with accelerator
    model = accelerator.prepare(model)  # ADDED: Prepare the model using the accelerator's prepare function. This step ensures that the model is compatible with mixed-precision training and distributed training if applicable.
    return model

# Prepare data_collator with accelerator
data_collator = accelerator.prepare(data_collator)  # ADDED: Also prepare the data_collator using the accelerator. This step ensures that the data collator is compatible with mixed-precision and distributed training, which is important for efficient training on modern hardware.

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="llama7b-finetuned-xsum", evaluation_strategy="epoch",
    weight_decay=0.01, save_total_limit=3, num_train_epochs=4,
    predict_with_generate=True, fp16=True, push_to_hub=False)

# Trainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Hyperparameter search with Optuna
best_trial = trainer.hyperparameter_search(
    direction="maximize", backend="optuna", hp_space=optuna_hp_space, n_trials=20)

# Final training with best hyperparameters
training_args.learning_rate = best_trial.params["learning_rate"]
training_args.per_device_train_batch_size = best_trial.params["per_device_train_batch_size"]
trainer.train()
```
