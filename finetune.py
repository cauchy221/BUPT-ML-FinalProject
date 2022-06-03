from distutils.command.config import config
import logging
import torch
from torch.utils.data import DataLoader
import os
import argparse
import datasets
from datasets import Dataset, load_metric
import transformers
from transformers import (SchedulerType, AutoConfig, BertTokenizer,
                          BertForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler)
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
import socket
import math
from tqdm.auto import tqdm
import json
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a model for sentiment analysis")
    # data
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dire", type=str, default="output/example")
    parser.add_argument("--train_file", type=str,
                        default="data/preprocessed/train.csv")
    parser.add_argument("--validation_file", type=str,
                        default="data/preprocessed/dev.csv")
    parser.add_argument("--test_file", type=str,
                        default="data/preprocessed/test.csv")
    parser.add_argument("--max_source_length", type=int, default=512)
    # params
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", choices=[
                        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=str, default="epoch",
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    # wandb
    parser.add_argument("--team_name", type=str, default="iriss")
    parser.add_argument("--project_name", type=str, default="ML FinalProject")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--scenario_name", type=str)

    args = parser.parse_args()
    return args


def load_data(data_path):
    if 'test' in data_path:
        results = {'id': [], 'text': []}
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                line_lst = line[:-1].split(',')
                results["id"].append(line_lst[0])
                results["text"].append(line_lst[1])
            results = Dataset.from_dict(results)
    else:
        results = {'sentiment': [], 'text': []}
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                line_lst = line[:-1].split(',')
                results["sentiment"].append(line_lst[0])
                results["text"].append(line_lst[1])
            results = Dataset.from_dict(results)
    return results


def load_datasets(args):
    datasets = {}
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    for key in data_files:
        datasets[key] = load_data(data_files[key])
    return datasets


def main():
    # args
    args = parse_args()

    # accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if args.output_dire is not None:
            os.makedirs(args.output_dire, exist_ok=True)
    accelerator.wait_for_everyone()

    # wandb
    wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name+" seed:"+str(args.seed),
               group=args.scenario_name,
               dir=str(args.output_dire),
               job_type="training",
               reinit=True,
               )

    # log
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    # datasets
    raw_datasets = load_datasets(args)

    # model
    config = AutoConfig.from_pretrained(args.model_path, num_labels=5)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(
        args.model_path, config=config, ignore_mismatched_sizes=True)

    # data preprocess
    def preprocess_function(examples):
        inputs = examples['text']
        print(inputs)
        labels = examples['sentiment']
        print(labels)
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, truncation=True, padding=True)
        model_inputs["labels"] = [int(l) for l in labels]
        return model_inputs

    # dataset
    with accelerator.main_process_first():
        train_dataset = raw_datasets['train']
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
        )
        # train_dataset = train_dataset.remove_columns('token_type_ids')
        # eval_dataset = eval_dataset.remove_columns('token_type_ids')

    # data collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # data loader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_batch_size)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # scheduler and math
    num_update_steps_per_epoch = math.ceil(len(
        train_dataloader) / args.gradient_accumulation_steps / accelerator.num_processes)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # save states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # metric
    metric = load_metric('accuracy')

    # train and eval
    total_batch_size = args.per_device_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        # train
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs['loss']
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                wandb.log({'loss': loss}, step=completed_steps)
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dire = f"step_{completed_steps}"
                    if args.output_dire is not None:
                        output_dire = os.path.join(args.output_dire, output_dire)
                    # accelerator.save_state(output_dire)
            if completed_steps >= args.max_train_steps:
                break

        # eval
        model.eval()
        samples_seen = 0
        eval_preds = []
        eval_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(**batch)
            preds = outputs.logits.argmax(dim=-1)
            preds, refs = accelerator.gather((preds, batch['labels']))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    preds = preds[: len(
                        eval_dataloader.dataset) - samples_seen]
                    refs = refs[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += refs.shape[0]
            eval_preds.extend(preds)
            eval_labels.extend(refs)
            metric.add_batch(predictions=preds, references=refs)

        eval_metric = metric.compute()
        result = {'epoch': epoch, 'acc': eval_metric}
        logger.info(result)
        wandb.log(result)

        eval_output = []
        for i in range(0, len(eval_preds)):
            eval_output.append(eval_preds[i] + '\t' + eval_labels[i])

        # test
        test_output = []
        for src in raw_datasets['test']['text']:
            with torch.no_grad():
                input_ids = tokenizer.encode(
                    src, max_length=args.max_source_length, truncation=True, return_tensors='pt')
                preds = accelerator.unwrap_model(model).generate(
                    input_ids.cuda()).logits.argmax(dim=-1)
            test_output.append(preds+1)  # 最终输出1-5类

        if args.checkpointing_steps == "epoch":
            output_dire = f"epoch_{epoch}"
            if args.output_dire is not None:
                output_dire = os.path.join(args.output_dire, output_dire)
            accelerator.save_state(output_dire)
            with open(os.path.join(output_dire, 'eval_output.csv'), "w", encoding='UTF-8') as writer:
                writer.write("\n".join(eval_output))
            with open(os.path.join(output_dire, 'test_output.csv'), "w", encoding='UTF-8') as writer:
                writer.write("\n".join(test_output))

    # all finish
    wandb.finish()
    if args.output_dire is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dire, save_function=accelerator.save)
        print("save model")
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dire)
        with open(os.path.join(args.output_dire, "all_results.json"), "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
