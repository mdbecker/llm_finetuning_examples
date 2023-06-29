## Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA)

The modified script incorporates Parameter-Efficient Fine-Tuning (PEFT) using the Low-Rank Adaptation (LoRA) technique. This is particularly useful for fine-tuning large language models with significantly fewer parameters, reducing computational and storage costs. Here are some key insights and benefits of using LoRA:

### Benefits of Using LoRA

LoRA enables efficient adaptation of pre-trained models by fine-tuning a subset of the model's parameters. This can be crucial for adapting large language models in resource-constrained environments without compromising much on performance. Some benefits include:

- **Resource Efficiency**: PEFT is beneficial when working on consumer-grade hardware that does not have the computational capabilities to handle the full fine-tuning of large language models.

- **Edge Deployment**: LoRA is suitable for deploying models on edge devices where storage and computation are limited.

- **Rapid Prototyping**: LoRA is helpful for experimenting and needing quick iterations without the full expense of time and computation.

- **Cost Reduction**: In cloud computing environments, reducing computation can have a direct impact on cost.

### Configuring LoRA

- `task_type`: The task type for which the model is being fine-tuned is specified. In this script, it is set to Sequence-to-Sequence Language Modeling (`SEQ_2_SEQ_LM`).

- `r`: This is the rank of the low-rank transformation. It determines the size of the projection matrix and is a hyperparameter in LoRA.

- `lora_alpha`: This is the size of the hidden low-rank factors. It affects the rank of the projection matrix.

- `lora_dropout`: Dropout rate for regularization in LoRA layers.

### Adapting the Model for PEFT

The model is adapted for PEFT by wrapping it with the `get_peft_model` function, which takes the original model and the LoRA configuration as inputs. This essentially applies the LoRA transformation to the model.

### Other Parameter-Efficient Fine-Tuning (PEFT) Techniques Supported by the PEFT Library

Apart from LoRA, the PEFT library supports various other methods to efficiently fine-tune pre-trained language models. Below is a summary of these methods:

1. **Prefix Tuning**: This technique adds a task-specific learnable prefix to the input sequence. This prefix helps in guiding the model's generative capabilities towards a specific task.

   Example code:
   ```python
   from peft import get_peft_model, PrefixTuningConfig, TaskType
   
   prefix_tuning_config = PrefixTuningConfig(
       task_type=TaskType.CAUSAL_LM,
       inference_mode=False,
       prefix_dropout=0.1,
       prefix_init_style="constant",
       prefix_init_token_id=50256
   )
   
   model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
   model = get_peft_model(model, prefix_tuning_config)
   ```

2. **P-Tuning**: Similar to Prefix Tuning, P-Tuning introduces prompt tokens, which are learnable tokens placed at the beginning of the input sequence. These prompt tokens help in task-specific tuning.

   Example code:
   ```python
   from peft import get_peft_model, PromptEncoderConfig
   
   prompt_encoder_config = PromptEncoderConfig(
       task_type="SEQ_CLS",
       num_virtual_tokens=20,
       encoder_hidden_size=128
   )
   
   model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
   model = get_peft_model(model, prompt_encoder_config)
   ```

3. **Prompt Tuning**: It is another variation of adding learnable tokens (prompts) to the input, where the prompts are typically larger and optimized to guide the model for a particular task.

   Example code:
   ```python
   from peft import get_peft_model, PromptTuningConfig, TaskType
   
   prompt_tuning_config = PromptTuningConfig(
       task_type=TaskType.SEQ_CLS,
       inference_mode=False,
       num_virtual_tokens=20,
       prompt_tuning_init_style="constant",
       prompt_tuning_init_token_id=50256
   )
   
   model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
   model = get_peft_model(model, prompt_tuning_config)
   ```

4. **AdaLoRA**: This technique combines the LoRA approach with adaptive budget allocation, which allows the method to adjust the number of parameters fine-tuned depending on resource constraints.

   Example code:
   ```python
   from peft import get_peft_model, AdaLoraConfig, TaskType
   
   adalora_config = AdaLoraConfig(
       task_type=TaskType.SEQ_CLS,
       inference_mode=False,
       lora_alpha=32,
       lora_dropout=0.1,
       base_model_alpha=0.9,
       ada_alpha=0.5,
       ada_max_step=20,
       ada_min_step=10,
   )
   
   model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
   model = get_peft_model(model, adalora_config)
   ```

5. **LLaMA-Adapter**: This method focuses on fine-tuning language models with zero-initialized attention, which leads to efficiency in the training process.

6. **INT8 Training with PEFT LoRA and Bits and Bytes**: This technique is for training large models using INT8 precision on platforms like Google Colab. This is especially useful for reducing memory footprint and accelerating training on hardware that supports INT8 operations.

   Example code:
   ```python
   import bitsandbytes as bnb


   from peft import LoraConfig, get_peft_model
   
   # Other configuration and model initialization code...
   
   config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   
   model = get_peft_model(model, config)
   
   # Training and inference steps
   # ...
   ```

These methods enable efficient fine-tuning of language models with fewer parameters, which is beneficial for reducing computational and storage costs.

### Diff
The difference between the original code [base.md](base.md) and this new version are:

```diff
@@ -2,6 +2,7 @@ import numpy as np
 import optuna
 from datasets import load_dataset, load_metric
 from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
+from peft import get_peft_model, LoraConfig, TaskType  # MODIFIED: Importing necessary components for PEFT
 
 # Data and model loading
 raw_datasets = load_dataset("xsum")
@@ -36,13 +37,22 @@ def optuna_hp_space(trial):
         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
     }
 
-# Model initialization function for Optuna
+# PEFT Configuration  # MODIFIED
+peft_config = LoraConfig(
+    task_type=TaskType.SEQ_2_SEQ_LM,   # MODIFIED: Specifying the task type for PEFT as sequence-to-sequence language modeling.
+    inference_mode=False,              # MODIFIED: Disabling inference mode during training.
+    r=8,                               # MODIFIED: Setting the reduction factor for PEFT. It determines the size of the projection matrix.
+    lora_alpha=32,                     # MODIFIED: Setting the alpha parameter, which affects the projection matrix's rank.
+    lora_dropout=0.1                   # MODIFIED: Setting the dropout rate for the LoRA layers.
+)
+
+# Model initialization function for Optuna and PEFT  # MODIFIED
 def model_init(trial):
     model = LlamaForCausalLM.from_pretrained(model_checkpoint)
     # handle pad_token issue
     if padding_added:
         model.resize_token_embeddings(len(tokenizer))
-    return model
+    return get_peft_model(model, peft_config)  # MODIFIED: Wrapping the model with PEFT by applying the LoRA config.
 
 # Training arguments
 training_args = Seq2SeqTrainingArguments(
```

### Full file:
```python
import numpy as np
import optuna
from datasets import load_dataset, load_metric
from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType  # MODIFIED: Importing necessary components for PEFT

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

# PEFT Configuration  # MODIFIED
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,   # MODIFIED: Specifying the task type for PEFT as sequence-to-sequence language modeling.
    inference_mode=False,              # MODIFIED: Disabling inference mode during training.
    r=8,                               # MODIFIED: Setting the reduction factor for PEFT. It determines the size of the projection matrix.
    lora_alpha=32,                     # MODIFIED: Setting the alpha parameter, which affects the projection matrix's rank.
    lora_dropout=0.1                   # MODIFIED: Setting the dropout rate for the LoRA layers.
)

# Model initialization function for Optuna and PEFT  # MODIFIED
def model_init(trial):
    model = LlamaForCausalLM.from_pretrained(model_checkpoint)
    # handle pad_token issue
    if padding_added:
        model.resize_token_embeddings(len(tokenizer))
    return get_peft_model(model, peft_config)  # MODIFIED: Wrapping the model with PEFT by applying the LoRA config.

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
