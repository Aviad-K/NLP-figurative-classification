import os
import json
import shutil
import torch
import numpy as np
import pandas as pd
import glob
import argparse
import nltk

from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
from torch.profiler import profile, record_function, ProfilerActivity

# --- Model Definition ---

class RobertaForTokenClassificationWithAllPOS(RobertaPreTrainedModel):
    def __init__(self, config, pos_vocab_size, fgpos_vocab_size, pos_embedding_dim, fgpos_embedding_dim):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        self.pos_embedding_dim = pos_embedding_dim
        self.fgpos_embedding_dim = fgpos_embedding_dim

        self.pos_embedding = torch.nn.Embedding(pos_vocab_size, self.pos_embedding_dim)
        
        classifier_input_size = config.hidden_size + self.pos_embedding_dim

        if self.fgpos_embedding_dim > 0:
            self.fgpos_embedding = torch.nn.Embedding(fgpos_vocab_size, self.fgpos_embedding_dim)
            classifier_input_size += self.fgpos_embedding_dim
        else:
            self.fgpos_embedding = None

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(classifier_input_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pos_tag_ids=None,
        fgpos_tag_ids=None,
        pos_attention_mask=None,
        fgpos_attention_mask=None,
        labels=None,
        **kwargs
    ):
        roberta_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = roberta_output[0]

        pos_embeddings = self.pos_embedding(pos_tag_ids)
        pos_embeddings = pos_embeddings * pos_attention_mask.unsqueeze(-1)

        outputs_to_combine = [sequence_output, pos_embeddings]

        if self.fgpos_embedding is not None and self.fgpos_embedding_dim > 0:
            fgpos_embeddings = self.fgpos_embedding(fgpos_tag_ids)
            fgpos_embeddings = fgpos_embeddings * fgpos_attention_mask.unsqueeze(-1)
            outputs_to_combine.append(fgpos_embeddings)

        combined_output = torch.cat(outputs_to_combine, dim=-1)
        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=roberta_output.hidden_states,
            attentions=roberta_output.attentions,
        )

# --- Trainer and Metrics ---

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    predicted_labels = []
    for prediction, label in zip(predictions, labels):
        for p_val, l_val in zip(prediction, label):
            if l_val != -100:
                true_labels.append(l_val)
                predicted_labels.append(p_val)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', pos_label=1, zero_division=0
    )
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store weights on CPU, move to device in compute_loss
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Move weights to the same device as logits inside compute_loss
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Helper Functions ---

def load_and_group_data_pandas(json_path):
    """
    Loads and processes data using pandas.
    IMPORTANT: We separate *occurrences* of identical sentence strings so they don't get merged.
    Output keys and formats are identical to your original function.
    """
    df = pd.read_json(json_path, lines=True, encoding='utf-8')

    # Sanity: require the columns your pipeline already uses
    required = ["sentence", "w_index", "label", "POS", "FGPOS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {json_path}: {missing}")

    # Preserve original file order and create a per-sentence occurrence id.
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))

    # Sort to keep tokens in order per sentence and preserve file order across repeats
    df_sorted = df.sort_values(by=["sentence", "_row", "w_index"])

    # New: split identical sentence strings into separate occurrences
    # Start of a sentence occurrence is marked by w_index == 0
    df_sorted["_is_start"] = (df_sorted["w_index"] == 0).astype(int)
    # Occurrence id: cumulative count of starts within each sentence string
    df_sorted["_occ"] = df_sorted.groupby("sentence")["_is_start"].cumsum() - 1

    # Group by (sentence, occurrence), not just sentence
    grouped = df_sorted.groupby(["sentence", "_occ"], sort=False).agg({
        "w_index": list,
        "label": list,
        "POS": list,
        "FGPOS": list
    }).reset_index()

    grouped_data = []
    for _, row in grouped.iterrows():
        # Reconstruct tokens exactly as your original code did
        original_words = row["sentence"].split(' ')
        try:
            words_for_model = [original_words[i] for i in row["w_index"]]
        except IndexError:
            # Guard against any stray bad index; skip that bad sentence cleanly
            print(f"Skipping problematic sentence due to w_index mismatch: {row['sentence']}")
            continue

        grouped_data.append({
            "sentence_words": words_for_model,   # list[str]
            "labels": row["label"],              # list[int]
            "pos_tags": row["POS"],              # list[str]
            "fgpos_tags": row["FGPOS"],          # list[str]
            # Keep the exact original sentence string for clean CSVs
            "original_sentence": row["sentence"]
        })

    return grouped_data


def tokenize_and_align_labels(examples, tokenizer, pos2id, fgpos2id):
        """
        A truly vectorized function to tokenize and align labels for a batch of examples.
        This version avoids all Python loops and list comprehensions over examples, 
        using NumPy array operations for maximum performance with `datasets.map(batched=True)`.
        """
        # Tokenize the batch of sentences. `is_split_into_words=True` is crucial.
        tokenized_inputs = tokenizer(
            examples["sentence_words"],
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True,
        )

        # This is one of the few necessary list comprehensions as word_ids are generated per example.
        all_word_ids_list = [tokenized_inputs.word_ids(i) for i in range(len(examples["sentence_words"]))]

        # --- Vectorized Data Preparation ---

        # Flatten all labels and tags from the batch into 1D arrays. This is much faster than looping.
        flat_labels = np.concatenate(examples["labels"])
        flat_pos_tags = np.concatenate([np.array([pos2id[tag] for tag in tags]) for tags in examples["pos_tags"]])
        flat_fgpos_tags = np.concatenate([np.array([fgpos2id[tag] for tag in tags]) for tags in examples["fgpos_tags"]])

        # Get sentence lengths and compute cumulative offsets for flattened arrays.
        sentence_lengths = np.array([len(s) for s in examples["sentence_words"]])
        sentence_offsets = np.concatenate(([0], np.cumsum(sentence_lengths[:-1])))

        # --- Vectorized Label and Tag Alignment ---

        batch_size = len(all_word_ids_list)
        max_len = 128
        
        # Initialize output arrays
        labels = np.full((batch_size, max_len), -100, dtype=np.int64)
        pos_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        fgpos_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        pos_attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        fgpos_attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        # Create arrays for row indices and word_ids, handling None for special tokens
        row_indices = np.arange(batch_size).repeat(max_len)
        word_ids_flat = np.array([w if w is not None else -1 for w_list in all_word_ids_list for w in w_list])
        
        # Create a mask for valid tokens (not special tokens)
        valid_token_mask = word_ids_flat != -1
        
        # Get the row and column indices for valid tokens
        valid_rows = row_indices[valid_token_mask]
        valid_cols = np.where(valid_token_mask)[0] % max_len
        
        # Get the corresponding word_ids and apply sentence offsets
        valid_word_ids = word_ids_flat[valid_token_mask]
        flat_indices = valid_word_ids + sentence_offsets[valid_rows]

        # Identify the first token of each word
        # Compare each word_id with the previous one in the sequence
        previous_word_ids = np.pad(word_ids_flat, (1, 0))[:-1]
        # A token is the first if it's valid and its word_id differs from the previous one
        # (or if it's the very first token of a sentence)
        is_first_token_mask = valid_token_mask & (word_ids_flat != previous_word_ids)
        
        # Get the row/col indices for only the first tokens
        first_token_rows = row_indices[is_first_token_mask]
        first_token_cols = np.where(is_first_token_mask)[0] % max_len
        
        # Get the flat indices for the first tokens
        first_token_flat_indices = valid_word_ids[is_first_token_mask[valid_token_mask]] + sentence_offsets[first_token_rows]

        # --- Vectorized Assignment ---
        # Assign labels and tags using the calculated indices. This is the core speed-up.
        labels[first_token_rows, first_token_cols] = flat_labels[first_token_flat_indices]
        pos_ids[first_token_rows, first_token_cols] = flat_pos_tags[first_token_flat_indices]
        fgpos_ids[first_token_rows, first_token_cols] = flat_fgpos_tags[first_token_flat_indices]
        pos_attention_mask[first_token_rows, first_token_cols] = 1
        fgpos_attention_mask[first_token_rows, first_token_cols] = 1

        tokenized_inputs["labels"] = labels.tolist()
        tokenized_inputs["pos_tag_ids"] = pos_ids.tolist()
        tokenized_inputs["fgpos_tag_ids"] = fgpos_ids.tolist()
        tokenized_inputs["pos_attention_mask"] = pos_attention_mask.tolist()
        tokenized_inputs["fgpos_attention_mask"] = fgpos_attention_mask.tolist()
        
        return tokenized_inputs

def list_of_dicts_to_dict_of_lists(d):
        if not d:
            return {}
        keys = d[0].keys()
        return {key: [item[key] for item in d] for key in keys}

def log_and_save_results(metrics, args, is_kfold=False):
    """Helper function to print and save evaluation results."""
    if not args.compute_metrics:
        return

    results_data = {}
    if is_kfold:
        if 'eval_f1' not in metrics:
            return
        
        print(f"\n--- K-Fold Cross-Validation Summary ---")
        mean_loss = np.mean(metrics['eval_loss'])
        mean_f1 = np.mean(metrics['eval_f1'])
        std_f1 = np.std(metrics['eval_f1'])
        mean_precision = np.mean(metrics['eval_precision'])
        mean_recall = np.mean(metrics['eval_recall'])

        print(f"Validation loss (mean): {mean_loss:.4f}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {mean_precision:.4f}")
        print(f"Recall: {mean_recall:.4f}")

        results_data = {
            "mean_eval_f1": mean_f1,
            "std_eval_f1": std_f1,
            "mean_eval_precision": mean_precision,
            "mean_eval_recall": mean_recall,
            "mean_eval_loss": mean_loss,
        }
    else:  # Single run
        print("\n--- Final Test Set Metrics ---")
        print(f"  F1: {metrics.get('eval_f1', -1):.4f}")
        print(f"  Precision: {metrics.get('eval_precision', -1):.4f}")
        print(f"  Recall: {metrics.get('eval_recall', -1):.4f}")
        print(f"  Accuracy: {metrics.get('eval_accuracy', -1):.4f}")

        results_data = {
            "test_f1": metrics.get('eval_f1'),
            "test_precision": metrics.get('eval_precision'),
            "test_recall": metrics.get('eval_recall'),
            "test_accuracy": metrics.get('eval_accuracy'),
            "test_loss": metrics.get('eval_loss'),
        }

    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results_data, f)
        print(f"Final results saved to {args.results_file}")

# --- Train and Evaluate Helper Function ---

def train_and_evaluate(args, train_dataset, eval_dataset, model_name, data_collator, pos_vocab_size, fgpos_vocab_size, device, output_dir, fold_info="", save_strategy="epoch", load_best_model_at_end=False, profile_run=False, seed=42):
    """
    Helper function to encapsulate a single training and evaluation run.
    """
    print(f"\n--- Starting {fold_info} ---")

    # --- Calculate Class Weights ---
    labels_flat = [label for sublist in train_dataset['labels'] for label in sublist if label != -100]
    counts = Counter(labels_flat)
    if len(counts) < 2:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
    else:
        total = sum(counts.values())
        weight_0 = total / counts.get(0, 1)
        weight_1 = total / counts.get(1, 1)
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

    # --- Initialize Model ---
    model = RobertaForTokenClassificationWithAllPOS.from_pretrained(
        model_name,
        num_labels=2,
        pos_vocab_size=pos_vocab_size,
        fgpos_vocab_size=fgpos_vocab_size,
        pos_embedding_dim=args.pos_embedding_dim,
        fgpos_embedding_dim=args.fgpos_embedding_dim,
    ).to(device)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=1,
        load_best_model_at_end=load_best_model_at_end,
        logging_steps=100,
        seed=seed,
        fp16=args.fp16,
        bf16=args.bf16,
        torch_compile=args.torch_compile,
        optim=args.optim,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        no_cuda=not (device.type == 'cuda'),
    )

    # --- Initialize Trainer ---
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if args.compute_metrics else None,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    # --- Train and Evaluate ---
    if profile_run:
        print("\n--- Profiling Run ---")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_training"):
                trainer.train()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    else:
        trainer.train()

    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    print(f"Saved model to {output_dir}")
    
    return metrics

# --- Error Analysis Function ---
def generate_error_analysis_file(original_test_data, tokenized_test_dataset, final_predictions, output_file):
    """
    Generates a CSV for error analysis with sentence-aligned predictions.
    Uses original_sentence (exact string from the dataset) for the CSV.
    """
    print(f"Generating error analysis file at: {output_file}")

    analysis_records = []

    predictions = np.array(final_predictions)                 # [num_sents, max_len]
    labels = np.array(tokenized_test_dataset["labels"])       # [num_sents, max_len]

    # Basic shape sanity
    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")

    for sent_idx, sentence_data in enumerate(original_test_data):
        # Exact sentence text as it appeared in the JSONL
        full_sentence = sentence_data.get(
            "original_sentence",
            " ".join(sentence_data["sentence_words"])
        )

        pred_sentence = predictions[sent_idx]
        label_sentence = labels[sent_idx]

        # First-token positions (where you placed gold labels during alignment)
        first_token_positions = np.where(np.array(label_sentence) != -100)[0]

        # If the model truncated, limit to available words
        n_words = min(len(first_token_positions), len(sentence_data["sentence_words"]))

        # Optional safety: skip pathologically short/long mismatches
        if n_words == 0:
            continue

        for j in range(n_words):
            tok_pos = first_token_positions[j]
            word = sentence_data["sentence_words"][j]
            true_label = int(label_sentence[tok_pos])
            predicted_label = int(pred_sentence[tok_pos])

            error_type = "Correct"
            if predicted_label != true_label:
                error_type = "False Positive" if predicted_label == 1 else "False Negative"

            analysis_records.append({
                "word": word,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "error_type": error_type,
                "pos_tag": sentence_data["pos_tags"][j],
                "fgpos_tag": sentence_data["fgpos_tags"][j],
                "sentence": full_sentence,
            })

    df = pd.DataFrame(analysis_records)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print("Error analysis file generated successfully.")


# --- Main Function ---

def main(args):
    # --- Performance Optimizations ---
    if args.use_flash_sdp and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("Flash Attention enabled.")
        except Exception as e:
            print(f"Could not enable Flash Attention: {e}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- CUDA and Device Setup ---
    if torch.cuda.is_available():
        print(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
        print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
    else:
        print("CUDA not available. Training on CPU.")
        device = torch.device("cpu")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, add_prefix_space="roberta" in args.model_name.lower())

    # --- Load and process data using pandas for performance ---
    print("Loading and processing data with pandas...")

    train_data = load_and_group_data_pandas(os.path.join(args.data_dir, "vua20_metaphor_train.json"))
    test_data = load_and_group_data_pandas(os.path.join(args.data_dir, "vua20_metaphor_test.json"))
    

    # --- Create Vocabularies ---
    all_pos_tags = sorted(list(set(tag for item in train_data for tag in item['pos_tags']).union(set(tag for item in test_data for tag in item['pos_tags']))))
    pos2id = {tag: i for i, tag in enumerate(all_pos_tags)}
    pos_vocab_size = len(pos2id)

    all_fgpos_tags = sorted(list(set(tag for item in train_data for tag in item['fgpos_tags']).union(set(tag for item in test_data for tag in item['fgpos_tags']))))
    fgpos2id = {tag: i for i, tag in enumerate(all_fgpos_tags)}
    fgpos_vocab_size = len(fgpos2id)

    print("Tokenizing and aligning datasets in-memory...")

    # Convert data to the format expected by the processing function
    train_data_dict = list_of_dicts_to_dict_of_lists(train_data)
    test_data_dict = list_of_dicts_to_dict_of_lists(test_data)

    # Process the entire dataset in-memory using the vectorized function
    tokenized_train_dict = tokenize_and_align_labels(train_data_dict, tokenizer, pos2id, fgpos2id)
    tokenized_test_dict = tokenize_and_align_labels(test_data_dict, tokenizer, pos2id, fgpos2id)

    # After building tokenized_test_dataset
    expected_words = sum(len(x) for x in test_data_dict["sentence_words"])
    observed_words = int((np.array(tokenized_test_dict["labels"]) != -100).sum())
    print("Words in test_data vs. first-token labels:", expected_words, observed_words)


    # Create Hugging Face Dataset objects from the in-memory dictionaries
    # The columns are already aligned, so no need to remove any.
    tokenized_train_dataset = Dataset.from_dict(tokenized_train_dict)
    tokenized_test_dataset = Dataset.from_dict(tokenized_test_dict)

    def sanity_check_samples(data, n=3):
        print("\n--- Sanity Check: Sample Sentences ---")
        for i, sample in enumerate(data[:n]):
            print(f"\nSentence {i+1}: {sample.get('original_sentence', ' '.join(sample['sentence_words']))}")
            print("Tokens : ", sample["sentence_words"])
            print("Labels : ", sample["labels"])
            print("POS    : ", sample["pos_tags"])
            print("FGPOS  : ", sample["fgpos_tags"])
            # Quick consistency checks
            assert len(sample["sentence_words"]) == len(sample["labels"]) == len(sample["pos_tags"]) == len(sample["fgpos_tags"]), \
                "Length mismatch detected!"
        print("\nSanity check passed for first", n, "examples (no mismatches).")
    # sanity_check_samples(train_data, n=100)

    def sanity_check_tokenized(original_data, tokenized_dataset, n=3):
        """
        Checks alignment after tokenization.
        Confirms number of non -100 labels == number of words per sentence (unless truncated).
        """
        print("\n--- Sanity Check: Tokenized Samples (Post-tokenization) ---")
        labels = np.array(tokenized_dataset["labels"])
        for i, sample in enumerate(original_data[:n]):
            num_words = len(sample["sentence_words"])
            num_labels = int((labels[i] != -100).sum())
            print(f"Sentence {i+1}:")
            print("  Words in original   :", num_words)
            print("  Non -100 labels     :", num_labels)
            if num_words != num_labels:
                print("  ⚠️ Mismatch! (likely due to truncation at max_length)")
            else:
                print("  ✅ Match")
        print("Post-tokenization sanity check done.\n")
    # sanity_check_tokenized(train_data, tokenized_train_dataset, n=100)

    
    print("Tokenizing complete. Data is now fully processed in memory.")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    if args.do_train:
        # If n_splits > 1, perform k-fold cross-validation.
        # Otherwise, perform a single training run on the full dataset.
        if args.n_splits > 1:
            print(f"Starting {args.n_splits}-fold cross-validation training...")
            kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            fold_metrics = defaultdict(list)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(tokenized_train_dataset)):
                print(f"\n=== Fold {fold_idx + 1}/{args.n_splits} ===")
                
                train_dataset_fold = tokenized_train_dataset.select(train_idx)
                val_dataset_fold = tokenized_train_dataset.select(val_idx)

                fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx + 1}')
                fold_info = f"Fold {fold_idx + 1}/{args.n_splits}"
                
                metrics = train_and_evaluate(
                    args=args,
                    train_dataset=train_dataset_fold,
                    eval_dataset=val_dataset_fold,
                    model_name=args.model_name,
                    data_collator=data_collator,
                    pos_vocab_size=pos_vocab_size,
                    fgpos_vocab_size=fgpos_vocab_size,
                    device=device,
                    output_dir=fold_output_dir,
                    fold_info=fold_info,
                    save_strategy="no", # Don't save checkpoints during k-fold
                    profile_run=(fold_idx == 0 and args.profile),
                    seed=args.seed + fold_idx
                )

                fold_metrics["eval_loss"].append(metrics["eval_loss"])
                if args.compute_metrics:
                    fold_metrics["eval_f1"].append(metrics["eval_f1"])
                    fold_metrics["eval_precision"].append(metrics["eval_precision"])
                    fold_metrics["eval_recall"].append(metrics["eval_recall"])

            log_and_save_results(fold_metrics, args, is_kfold=True)
        
        else: # This block handles the final training run without folds
            print("Starting final training run without cross-validation...")
            
            # Use the full training dataset and the test dataset for evaluation
            train_dataset_final = tokenized_train_dataset
            eval_dataset_final = tokenized_test_dataset

            metrics = train_and_evaluate(
                args=args,
                train_dataset=train_dataset_final,
                eval_dataset=eval_dataset_final,
                model_name=args.model_name,
                data_collator=data_collator,
                pos_vocab_size=pos_vocab_size,
                fgpos_vocab_size=fgpos_vocab_size,
                device=device,
                output_dir=args.output_dir,
                fold_info="Final Training Run",
                save_strategy="epoch",
                load_best_model_at_end=True,
                seed=args.seed
            )
            
            log_and_save_results(metrics, args, is_kfold=False)


    if args.do_eval:
        print("\n--- Starting Ensemble Evaluation on Test Set ---")
        model_dirs = sorted(glob.glob(os.path.join(args.output_dir, "fold_*")))
        models = []
        for d in model_dirs:
            if os.path.exists(os.path.join(d, "pytorch_model.bin")) or os.path.exists(os.path.join(d, "model.safetensors")):
                model = RobertaForTokenClassificationWithAllPOS.from_pretrained(
                    d,
                    pos_vocab_size=pos_vocab_size,
                    fgpos_vocab_size=fgpos_vocab_size,
                    pos_embedding_dim=args.pos_embedding_dim,
                    fgpos_embedding_dim=args.fgpos_embedding_dim
                )
                models.append(model)
        
        if not models:
            print("No models found for evaluation. Please run training first.")
            return

        print(f"Loaded {len(models)} models for ensemble prediction.")

        eval_args = TrainingArguments(
            output_dir="./inference_tmp", 
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        )
        predictor = Trainer(model=models[0], args=eval_args, data_collator=data_collator)

        per_model_logits = []
        for model in models:
            predictor.model = model.to(predictor.args.device)
            pred_out = predictor.predict(tokenized_test_dataset)
            per_model_logits.append(pred_out.predictions)

        # --- Soft Voting: Average logits instead of hard predictions ---
        # Stack the logits and compute the mean across the models (axis=0)
        mean_logits = np.mean(per_model_logits, axis=0)
        # Get the final predictions from the averaged logits
        y_pred_final = np.argmax(mean_logits, axis=-1)

        labels = np.array(tokenized_test_dataset['labels'])
        mask = labels != -100
        y_true = labels[mask]
        y_pred_masked = y_pred_final[mask]

        print("\n--- Evaluating Soft Voting Ensemble Performance ---")
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_masked, average="binary", pos_label=1, zero_division=0
        )
        acc = accuracy_score(y_true, y_pred_masked)

        print(f"Precision | Recall    | F1-Score  | Accuracy")
        print("-------------------------------------------------")
        print(f"{prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {acc:<9.4f}")

        # --- Generate Error Analysis File ---  <-- ADD THIS BLOCK
        if args.analysis_file:
            generate_error_analysis_file(
                original_test_data=test_data,
                tokenized_test_dataset=tokenized_test_dataset,
                final_predictions=y_pred_final,
                output_file=args.analysis_file
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a token classification model.")
    
    # --- Paths and Flags ---
    parser.add_argument("--data_dir", type=str, default="vua_dataset", help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, default="results_with_all_pos_reduced_dim_compiled", help="Directory to save models and results.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the pretrained model to use.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run the training loop.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run the ensemble evaluation.")
    parser.add_argument("--analysis_file", type=str, default=None, help="Path to save the error analysis CSV file.")

    # --- Model Hyperparameters ---
    parser.add_argument("--pos_embedding_dim", type=int, default=8, help="Dimension of the POS embeddings.")
    parser.add_argument("--fgpos_embedding_dim", type=int, default=10, help="Dimension of the FGPOS embeddings.")

    # --- Training Hyperparameters ---
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for regularization.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for the first training fold.")
    parser.add_argument("--compute_metrics", action="store_true", help="Enable metric computation during evaluation.")
    parser.add_argument("--results_file", type=str, default=None, help="Path to save the final results JSON file.")

    # --- Performance Arguments ---
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer to use, e.g., 'adamw_torch'.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision training (requires Ampere GPU).")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for JIT compilation.")
    parser.add_argument("--no-flash-sdp", action="store_false", dest="use_flash_sdp", help="Disable Flash Attention.")
    parser.set_defaults(use_flash_sdp=True)
 
    args = parser.parse_args()

    # --- Set precision based on args ---
    if args.bf16:
        args.fp16 = False # Ensure only one precision is active
    elif not args.bf16:
        # Default to fp16 if CUDA is available and no precision is specified
        if torch.cuda.is_available():
            args.fp16 = True

    main(args)