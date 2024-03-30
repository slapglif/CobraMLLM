import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def train(model, config, train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=config.patience, verbose=True
    )
    scaler = GradScaler()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")

        optimizer.zero_grad()
        for i, (vision_input, text_input, labels) in enumerate(train_progress, start=1):
            vision_input = vision_input.to(device, non_blocking=True)
            text_input = text_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(vision_input, text_input)
                loss = criterion(outputs, labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if i % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_progress.set_postfix({"Loss": loss.item()})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation")

        with torch.no_grad():
            for vision_input, text_input, labels in val_progress:
                vision_input = vision_input.to(device, non_blocking=True)
                text_input = text_input.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(vision_input, text_input)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_progress.set_postfix({"Loss": loss.item()})

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"""Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"""
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.early_stopping_patience:
                print("Early stopping triggered. Training stopped.")
                break

    print("Training completed.")
