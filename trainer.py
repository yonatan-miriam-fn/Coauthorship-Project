import csv
from enum import IntEnum

from datasets import load_dataset
from pathlib import Path
from trl import SFTConfig, SFTTrainer
from unsloth import apply_chat_template, FastLanguageModel, standardize_sharegpt, to_sharegpt

if __name__ == "main":
    import argparse

    # create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", metavar="model", required=True, choices=[
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
    ], help="The base model to fine-tune. Must be a model that supports 4bit quantization")
    parser.add_argument("-r", "--rank", metavar="lora_rank", type=int, default=16,
                        help="The LoRA rank of the fine-tuning process")
    parser.add_argument("-a", "--alpha", metavar="lora_alpha", type=int, default=16,
                        help="The scaling value of the fine-tuning process")
    parser.add_argument("-lr", "--learning_rate", metavar="learning_rate", type=float, default=2e-4,
                        help="The learning rate during fine-tuning")
    parser.add_argument("-t", "--template", metavar="template_index", type=int, choices=[0, 1, 2],
                        default=0, help="The index of the chat template")
    parser.add_argument("-f", "-full", metavar="full_run", action="store_true",
                        help="If included, will attempt to go through a full epoch during training instead of 60 steps")
    args = parser.parse_args()

    # settings
    model_name = args.model
    # hyperparameters
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    learning_rate = 2e-4
    chat_template = """You are an expert forensic linguist.
    The following two segments of text are each taken from blog posts.
    Some blog post pairs come from the same author, others come from different authors.
    Your task is to determine whether these two posts were written by the same author or different authors."""
    if args.template_index == 0:
        chat_template += "    Focus on linguistic clues and writing patterns such as word usage, syntax, and capitalization, rather than semantic content."
    chat_template += """    Output \\'True\\' if they have the same author, or output \\'False\\' for different authors.
    Provide """
    if args.template_index == 2:
        chat_template += "one and only one verdict, and "
    chat_template += """no extraneous output.
        
    
    {INPUT}
    
    Output:
    {OUTPUT}"""


    # constants
    max_seq_length = 2048

    # load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True
    )

    # load dataset
    data_folder = Path("data") / "same-prior_0.5"
    dataset = load_dataset("csv", data_files={"train": str(data_folder / "train.csv"), "dev": str(data_folder / "dev.csv")})

    input_template = """Post #1:\n{Text A}\n\nPost #2:\n{Text B}"""

    train_dataset = to_sharegpt(
        dataset["train"],
        merged_prompt = input_template,
        output_column_name = "Same Author"
    )
    train_dataset = standardize_sharegpt(train_dataset)

    train_dataset = apply_chat_template(
        train_dataset,
        tokenizer = tokenizer,
        chat_template = chat_template
    )

    # apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 57205,
        use_rslora = False,
        loftq_config = None,
    )

    # create trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        max_seq_length = max_seq_length,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = -1 if args.full_run else 60,
            num_train_epochs = 1,
            learning_rate = learning_rate,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            seed = 52506,
            output_dir = "outputs",
            save_strategy = "steps",
            save_steps = 50
        ),
    )

    # train model
    trainer_stats = trainer.train()

    # save model
    save_name: str = "fine_tuned_model"
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

    # evaluate model
    class Prediction(IntEnum):
      TRUE = 0
      UNKNOWN = 1
      FALSE = 2

    def get_prediction(response: str) -> Prediction:
        matches_true = "true" in response.lower()
        matches_false = "false" in response.lower()
        if matches_true and not matches_false:
            return Prediction.TRUE
        elif matches_false and not matches_true:
            return Prediction.FALSE
        else:
            return Prediction.UNKNOWN

    FastLanguageModel.for_inference(model)

    # format: [[true positive, unknown (positive), false negative],
    #          [false positive, unknown (positive), true negative]]
    confusion_matrix = [[0, 0, 0], [0, 0, 0]]
    with open(data_folder / "dev.csv", "r", encoding="utf-8") as dev_file:
        reader = csv.DictReader(dev_file)

        for i in range(100):
            row = next(reader)
            prompt = f"Post #1:\n{row['Text A']}\n\nPost #2:\n{row['Text B']}"
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                return_tensors = "pt"
            ).to("cuda")

            output = model.generate(
                input_ids,
                max_new_tokens = 64,
                pad_token_id = tokenizer.eos_token_id)
            prediction = get_prediction("".join(tokenizer.batch_decode(
                output[:, input_ids.shape[1]:],
                skip_special_tokens = True
            )))

            gold_label = row["Same Author"]
            confusion_matrix[0 if gold_label == "True" else 1][prediction] += 1

            if i % 10 == 9:
                print(f"{i + 1} examples tested")

    print(f"After 100 examples, confusion matrix = {confusion_matrix}")