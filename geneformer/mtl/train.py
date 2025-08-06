import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .imports import *
from .model import GeneformerMultiTask
from .utils import calculate_task_specific_metrics, get_layer_freeze_range

from itertools import islice


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_wandb(config):
    if config.get("use_wandb", False):
        import wandb

        wandb.init(project=config["wandb_project"], config=config)
        print("Weights & Biases (wandb) initialized and will be used for logging.")
    else:
        print(
            "Weights & Biases (wandb) is not enabled. Logging will use other methods."
        )


def create_model(config, num_labels_list, device):
    model = GeneformerMultiTask(
        config["pretrained_path"],
        num_labels_list,
        dropout_rate=config["dropout_rate"],
        use_task_weights=config["use_task_weights"],
        task_weights=config["task_weights"],
        max_layers_to_freeze=config["max_layers_to_freeze"],
        use_attention_pooling=config["use_attention_pooling"],
    )
    if config["use_data_parallel"]:
        model = nn.DataParallel(model)
    return model.to(device)


def setup_optimizer_and_scheduler(model, config, total_steps):
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    if config["lr_scheduler_type"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif config["lr_scheduler_type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )

    return optimizer, scheduler


def train_epoch(
    model, train_loader, optimizer, scheduler, device, config, writer, epoch, trial=None, epoch_fraction=1.0
):
    model.train()
    total_batches = len(train_loader)
    # Determine how many batches to process for this (possibly fractional) epoch.
    target_batches = int(total_batches * epoch_fraction) if epoch_fraction < 1.0 else total_batches
    target_train_loader = list(islice(train_loader, target_batches))
    # progress_bar = tqdm(range(target_batches), desc=f"Epoch {epoch+1} (frac: {epoch_fraction}) / {config['epochs']}")
    progress_bar = tqdm(target_train_loader, desc=f"Epoch {epoch+1} (frac: {epoch_fraction}) / {config['epochs']}")
    for batch_idx, batch in enumerate(progress_bar):
        # if batch_idx > target_batches:
        #     break
        # batch = next(iter(train_loader))
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = [
            batch["labels"][task_name].to(device) for task_name in config["task_names"]
        ]

        loss, _, _ = model(input_ids, attention_mask, labels)
        loss.backward()

        if config["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        optimizer.step()
        scheduler.step()

        # Compute global progress in epochs (e.g., 0.5 means half an epoch completed)
        global_progress = epoch + (batch_idx + 1) / len(train_loader)

        # Only start reporting/pruning after reaching a minimum epoch progress.
        # Using config["min_epoch_for_pruning"] if specified, otherwise default to 0.5.
        min_prune_epoch = config.get("min_epoch_for_pruning", 0.5)
        if trial is not None and global_progress >= min_prune_epoch:
            trial.report(loss.item(), global_progress)
            if trial.should_prune():
                if config.get("use_wandb", False):
                    import wandb
                    wandb.finish()
                raise optuna.TrialPruned()
        
        writer.add_scalar(
            "Training Loss", loss.item(), epoch * len(train_loader) + batch_idx
        )
        if config.get("use_wandb", False):
            import wandb

            wandb.log({"Training Loss": loss.item()})

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return loss.item()  # Return the last batch loss


def validate_model(model, val_loader, device, config):
    model.eval()
    val_loss = 0.0
    task_true_labels = {task_name: [] for task_name in config["task_names"]}
    task_pred_labels = {task_name: [] for task_name in config["task_names"]}
    task_pred_probs = {task_name: [] for task_name in config["task_names"]}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = [
                batch["labels"][task_name].to(device)
                for task_name in config["task_names"]
            ]
            loss, logits, _ = model(input_ids, attention_mask, labels)
            val_loss += loss.item()

            for sample_idx in range(len(batch["input_ids"])):
                for i, task_name in enumerate(config["task_names"]):
                    true_label = batch["labels"][task_name][sample_idx].item()
                    pred_label = torch.argmax(logits[i][sample_idx], dim=-1).item()
                    pred_prob = (
                        torch.softmax(logits[i][sample_idx], dim=-1).cpu().numpy()
                    )
                    task_true_labels[task_name].append(true_label)
                    task_pred_labels[task_name].append(pred_label)
                    task_pred_probs[task_name].append(pred_prob)

    val_loss /= len(val_loader)
    return val_loss, task_true_labels, task_pred_labels, task_pred_probs


def log_metrics(task_metrics, val_loss, config, writer, epochs):
    for task_name, metrics in task_metrics.items():
        print(
            f"{task_name} - Validation F1 Macro: {metrics['f1']:.4f}, Validation Accuracy: {metrics['accuracy']:.4f}"
        )
        if config.get("use_wandb", False):
            import wandb

            wandb.log(
                {
                    f"{task_name} Validation F1 Macro": metrics["f1"],
                    f"{task_name} Validation Accuracy": metrics["accuracy"],
                }
            )

    writer.add_scalar("Validation Loss", val_loss, epochs)
    for task_name, metrics in task_metrics.items():
        writer.add_scalar(f"{task_name} - Validation F1 Macro", metrics["f1"], epochs)
        writer.add_scalar(
            f"{task_name} - Validation Accuracy", metrics["accuracy"], epochs
        )


def save_validation_predictions(
    val_cell_id_mapping,
    task_true_labels,
    task_pred_labels,
    task_pred_probs,
    config,
    trial_number=None,
):
    if trial_number is not None:
        trial_results_dir = os.path.join(config["results_dir"], f"trial_{trial_number}")
        os.makedirs(trial_results_dir, exist_ok=True)
        val_preds_file = os.path.join(trial_results_dir, "val_preds.csv")
    else:
        val_preds_file = os.path.join(config["results_dir"], "manual_run_val_preds.csv")

    rows = []
    for sample_idx in range(len(val_cell_id_mapping)):
        row = {"Cell ID": val_cell_id_mapping[sample_idx]}
        for task_name in config["task_names"]:
            row[f"{task_name} True"] = task_true_labels[task_name][sample_idx]
            row[f"{task_name} Pred"] = task_pred_labels[task_name][sample_idx]
            row[f"{task_name} Probabilities"] = ",".join(
                map(str, task_pred_probs[task_name][sample_idx])
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(val_preds_file, index=False)
    print(f"Validation predictions saved to {val_preds_file}")


def train_model(
    config,
    device,
    train_loader,
    val_loader,
    train_cell_id_mapping,
    val_cell_id_mapping,
    num_labels_list,
):
    set_seed(config["seed"])
    initialize_wandb(config)

    model = create_model(config, num_labels_list, device)
    # Parse epochs as a float and compute total steps based on fractional epochs.
    num_epochs = float(config["epochs"])
    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    log_dir = os.path.join(config["tensorboard_log_dir"], "manual_run")
    writer = SummaryWriter(log_dir=log_dir)

    # Calculate full epochs and any fractional part
    num_full_epochs = int(num_epochs)
    fraction_epoch = num_epochs - num_full_epochs

    epoch_progress = tqdm(range(num_full_epochs), desc="Training Full Epochs")
    for epoch in epoch_progress:
        last_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, writer, epoch, trial=None, epoch_fraction=1.0
        )
        epoch_progress.set_postfix({"last_loss": f"{last_loss:.4f}"})

    # Train a final partial epoch if fraction_epoch > 0
    if fraction_epoch > 0:
        last_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            writer,
            num_full_epochs,  # continue from the full epochs count
            trial=None,
            epoch_fraction=fraction_epoch  # process only a fraction of the epoch
        )
        print(f"Partial Epoch {fraction_epoch} - last_loss: {last_loss:.4f}")

    val_loss, task_true_labels, task_pred_labels, task_pred_probs = validate_model(
        model, val_loader, device, config
    )
    task_metrics = calculate_task_specific_metrics(task_true_labels, task_pred_labels)

    log_metrics(task_metrics, val_loss, config, writer, config["epochs"])
    writer.close()

    save_validation_predictions(
        val_cell_id_mapping, task_true_labels, task_pred_labels, task_pred_probs, config
    )

    if config.get("use_wandb", False):
        import wandb

        wandb.finish()

    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    return val_loss, model  # Return both the validation loss and the trained model


def objective(
    trial,
    train_loader,
    val_loader,
    train_cell_id_mapping,
    val_cell_id_mapping,
    num_labels_list,
    config,
    device,
):
    set_seed(config["seed"])  # Set the seed before each trial
    initialize_wandb(config)

    # Hyperparameters
    config["learning_rate"] = trial.suggest_float(
        "learning_rate",
        config["hyperparameters"]["learning_rate"]["low"],
        config["hyperparameters"]["learning_rate"]["high"],
        log=config["hyperparameters"]["learning_rate"]["log"],
    )
    config["warmup_ratio"] = trial.suggest_float(
        "warmup_ratio",
        config["hyperparameters"]["warmup_ratio"]["low"],
        config["hyperparameters"]["warmup_ratio"]["high"],
    )
    config["weight_decay"] = trial.suggest_float(
        "weight_decay",
        config["hyperparameters"]["weight_decay"]["low"],
        config["hyperparameters"]["weight_decay"]["high"],
    )
    config["dropout_rate"] = trial.suggest_float(
        "dropout_rate",
        config["hyperparameters"]["dropout_rate"]["low"],
        config["hyperparameters"]["dropout_rate"]["high"],
    )
    config["lr_scheduler_type"] = trial.suggest_categorical(
        "lr_scheduler_type", config["hyperparameters"]["lr_scheduler_type"]["choices"]
    )
    config["use_attention_pooling"] = trial.suggest_categorical(
        "use_attention_pooling", [False]
    )

    if config["use_task_weights"]:
        if type(config["hyperparameters"]["task_weights"]) == list:
            config["task_weights"] = config["hyperparameters"]["task_weights"]  
        else:
            config["task_weights"] = [
                trial.suggest_float(
                    f"task_weight_{i}",
                    config["hyperparameters"]["task_weights"]["low"],
                    config["hyperparameters"]["task_weights"]["high"],
                )
                for i in range(len(num_labels_list))
            ]
            if "priority" in config["hyperparameters"]["task_weights"]:
                pri_index = config["hyperparameters"]["task_weights"]["priority"]
                the_max = max(config["task_weights"])
                if config["task_weights"][pri_index] != the_max:
                    config["task_weights"][pri_index] = trial.suggest_float(
                        f"task_weight_{pri_index}",
                        the_max,
                        config["hyperparameters"]["task_weights"]["high"],
                    )
        weight_sum = sum(config["task_weights"])
        config["task_weights"] = [
            weight / weight_sum for weight in config["task_weights"]
        ]
    else:
        config["task_weights"] = None

    # Dynamic range for max_layers_to_freeze
    if type(config["hyperparameters"]["max_layers_to_freeze"]) is not dict:
        freeze_range = get_layer_freeze_range(config["pretrained_path"])
        config["max_layers_to_freeze"] = trial.suggest_int(
            "max_layers_to_freeze",
            freeze_range["min"],
            freeze_range["max"]
        )
    else:
        config["max_layers_to_freeze"] = trial.suggest_int(
            "max_layers_to_freeze",
            config["hyperparameters"]["max_layers_to_freeze"]["min"],
            config["hyperparameters"]["max_layers_to_freeze"]["max"]
        )

    model = create_model(config, num_labels_list, device)
    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    log_dir = os.path.join(config["tensorboard_log_dir"], f"trial_{trial.number}")
    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = float(config["epochs"])
    num_full_epochs = int(num_epochs)
    fraction_epoch = num_epochs - num_full_epochs
    
    for epoch in range(num_full_epochs):
        train_epoch(
            model, train_loader, optimizer, scheduler, device, config, writer, epoch, trial, epoch_fraction=1.0
        )

    # Run a final partial epoch if needed
    if fraction_epoch > 0:
        train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            writer,
            num_full_epochs,  # continue progress from full epochs
            trial,
            epoch_fraction=fraction_epoch  # Process only a fraction of the epoch
        )

    val_loss, task_true_labels, task_pred_labels, task_pred_probs = validate_model(
        model, val_loader, device, config
    )
    task_metrics = calculate_task_specific_metrics(task_true_labels, task_pred_labels)

    log_metrics(task_metrics, val_loss, config, writer, config["epochs"])
    writer.close()

    save_validation_predictions(
        val_cell_id_mapping,
        task_true_labels,
        task_pred_labels,
        task_pred_probs,
        config,
        trial.number,
    )

    trial.set_user_attr("model_state_dict", model.state_dict())
    trial.set_user_attr("task_weights", config["task_weights"])

    trial.report(val_loss, config["epochs"])

    if trial.should_prune():
        raise optuna.TrialPruned()

    if config.get("use_wandb", False):
        import wandb

        wandb.log(
            {
                "trial_number": trial.number,
                "val_loss": val_loss,
                **{
                    f"{task_name}_f1": metrics["f1"]
                    for task_name, metrics in task_metrics.items()
                },
                **{
                    f"{task_name}_accuracy": metrics["accuracy"]
                    for task_name, metrics in task_metrics.items()
                },
                **{
                    k: v
                    for k, v in config.items()
                    if k
                    in [
                        "learning_rate",
                        "warmup_ratio",
                        "weight_decay",
                        "dropout_rate",
                        "lr_scheduler_type",
                        "use_attention_pooling",
                        "max_layers_to_freeze",
                    ]
                },
            }
        )
        wandb.finish()

    return val_loss
