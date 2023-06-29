## Leveraging Multiple GPUs with Hugging Face's Accelerate Library

The modified script employs the Accelerate library to facilitate distributed training across multiple GPUs. This not only enhances the speed but also ensures efficient resource allocation, particularly memory.

### Key Components and Benefits:

- **Accelerate Library**: A high-level library developed by Hugging Face. It's designed for seamless distribution of training across multiple GPUs or TPUs with minimal code changes.

- **Accelerator Class**: The `Accelerator` class is pivotal to the distributed training process. It handles device placements and gradient computations during training across multiple devices.

- **Automatic Device Placement**: Through the `Accelerator`, the script can automatically place tensors on the correct device without explicitly specifying it.

- **DeepSpeed Integration**: The script incorporates DeepSpeed, an optimization library renowned for faster and more memory-efficient training of deep learning models. This is especially advantageous for training large models and datasets.

- **Efficient Resource Allocation**: With the Accelerate library, the script can automatically determine the largest batch size that fits in memory. This maximizes GPU utilization without encountering out-of-memory errors.

### When to Use:

1. **Speed and Efficiency**: It's especially beneficial for large datasets or complex models where distributed training can significantly reduce training times.

2. **Handling Large Models and Datasets**: When the model or dataset is too large for a single GPU’s memory, distributed training across multiple GPUs or TPUs can be an effective solution.

3. **Optimizing Resource Usage**: Using DeepSpeed, memory utilization is optimized, allowing for larger batch sizes and models.

### Prerequisites:

Ensure that the Accelerate library is installed (`pip install accelerate`) and that the necessary hardware (multiple GPUs) is available. When utilizing DeepSpeed, a configuration file (`ds_config.json`) or equivalent settings must be provided.

### Diff
The difference between the original code [base.md](base.md) and this new version are:

```diff
@@ -1,7 +1,11 @@
 import numpy as np
 import optuna
 from datasets import load_dataset, load_metric
 from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
+from accelerate import Accelerator # MODIFIED
 
 # Data and model loading
 raw_datasets = load_dataset("xsum")
@@ -44,11 +48,16 @@ def model_init(trial):
         model.resize_token_embeddings(len(tokenizer))
     return model
 
+# Utilizing Both GPUs
+accelerator = Accelerator() # MODIFIED
+device = accelerator.device # MODIFIED
+
 # Training arguments
 training_args = Seq2SeqTrainingArguments(
     output_dir="llama7b-finetuned-xsum", evaluation_strategy="epoch",
     weight_decay=0.01, save_total_limit=3, num_train_epochs=4,
-    predict_with_generate=True, fp16=True, push_to_hub=False)
+    predict_with_generate=True, fp16=True, push_to_hub=False,
+    deepspeed="ds_config.json") # MODIFIED
 
 # Trainer
 trainer = Seq2SeqTrainer(
@@ -58,8 +67,8 @@ trainer = Seq2SeqTrainer(
     eval_dataset=tokenized_datasets["validation"],
     data_collator=data_collator,
     tokenizer=tokenizer,
-    compute_metrics=compute_metrics
-)
+    compute_metrics=compute_metrics,
+    accelerator=accelerator) # MODIFIED
 
 # Hyperparameter search with Optuna
 best_trial = trainer.hyperparameter_search(
@@ -69,3 +78,12 @@ best_trial = trainer.hyperparameter_search(
 training_args.learning_rate = best_trial.params["learning_rate"]
 training_args.per_device_train_batch_size = best_trial.params["per_device_train_batch_size"]
 trainer.train()
```

### Full file:
```python
import numpy as np
import optuna
from datasets import load_dataset, load_metric
from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from accelerate import Accelerator # MODIFIED

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

# Model initialization function for Optuna
def model_init(trial):
    model = LlamaForCausalLM.from_pretrained(model_checkpoint)
    # handle pad_token issue
    if padding_added:
        model.resize_token_embeddings(len(tokenizer))
    return model

# Utilizing Both GPUs
accelerator = Accelerator() # MODIFIED
device = accelerator.device # MODIFIED

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="llama7b-finetuned-xsum", evaluation_strategy="epoch",
    weight_decay=0.01, save_total_limit=3, num_train_epochs=4,
    predict_with_generate=True, fp16=True, push_to_hub=False,
    deepspeed="ds_config.json") # MODIFIED

# Trainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    accelerator=accelerator) # MODIFIED

# Hyperparameter search with Optuna
best_trial = trainer.hyperparameter_search(
    direction="maximize", backend="optuna", hp_space=optuna_hp_space, n_trials=20)

# Final training with best hyperparameters
training_args.learning_rate = best_trial.params["learning_rate"]
training_args.per_device_train_batch_size = best_trial.params["per_device_train_batch_size"]
trainer.train()
```
