import os
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate

# ✅ Load tokenizer at global scope for multiprocessing
tokenizer = BartTokenizer.from_pretrained("./bart-finetuned-mediasum")

def preprocess_function(examples):
    inputs = examples["document"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

def main():
    print("✅ Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ✅ Load model
    model_path = "./bart-finetuned-mediasum"
    model = BartForConditionalGeneration.from_pretrained(model_path)

    # ✅ Load datasets
    mediasum_path = "D:/xampp/htdocs/AI summarizer/mediasum_local"
    mediasum_dataset = load_from_disk(mediasum_path)["train"]

    # ✅ Select only unseen data: index 10000+
    mediasum_eval = mediasum_dataset.select(range(10000, len(mediasum_dataset)))
    combined_eval = concatenate_datasets([mediasum_eval]).select(range(1000))  
    print(f"🔍 Evaluating on {len(combined_eval)} unseen examples")

    # ✅ Tokenize
    tokenized_eval = combined_eval.map(
        preprocess_function,
        batched=True,
        remove_columns=["document", "summary"],
        num_proc=4  # multiprocessing now works because tokenizer is global
    )

    # ✅ Setup trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        report_to="none"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics
    )

    # ✅ Run evaluation
    metrics = trainer.evaluate()
    print("\n📊 ROUGE Evaluation Results on Unseen Data:")
    for key, value in metrics.items():
        if key.startswith("eval_"):
            print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()
