# enhanced_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from enhanced_maple import EnhancedMaPLe, EnhancedLoss, EnhancedConfig
from data_manager import get_dataloaders
from utils import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def train_enhanced():
    cfg = EnhancedConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create enhanced model
    model = EnhancedMaPLe(
        input_dim=28 * 28,
        num_classes=10,
        prompt_dim=cfg.prompt_dim,
        mask_ratio=cfg.mask_ratio,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
        use_attention=cfg.use_attention,
        use_residual=cfg.use_residual,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Enhanced loss function
    criterion = EnhancedLoss()

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Learning rate scheduler
    scheduler = StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    )

    # Data loaders
    train_loader, test_loader = get_dataloaders(cfg.batch_size)

    # Training history
    train_losses = []
    test_accuracies = []
    learning_rates = []

    best_accuracy = 0.0
    patience = 5
    patience_counter = 0

    print("Starting enhanced training...")

    for epoch in range(cfg.epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, training=True)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy
            if isinstance(outputs, tuple):
                predictions = outputs[0].argmax(dim=1)
            else:
                predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == y).sum().item()
            total_samples += y.size(0)

            # Update progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{current_acc:.4f}",
                    "LR": f'{optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

        # Evaluation phase
        model.eval()
        test_acc = evaluate_enhanced(model, test_loader, device)

        # Update scheduler
        scheduler.step()

        # Record metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / total_samples
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)
        learning_rates.append(current_lr)

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping and model saving
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                    "config": cfg,
                },
                "enhanced_maple_best.pth",
            )
            print(f"💾 New best model saved! Accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"⏰ Early stopping triggered after {patience} epochs without improvement"
            )
            break

    # Save final model
    torch.save(model.state_dict(), "enhanced_maple_final.pth")

    # Plot training curves
    plot_training_curves(train_losses, test_accuracies, learning_rates)

    print(f"🎉 Training completed! Best accuracy: {best_accuracy:.4f}")

    return model, best_accuracy


def evaluate_enhanced(model, test_loader, device):
    """Enhanced evaluation function"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, training=False)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


def plot_training_curves(train_losses, test_accuracies, learning_rates):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    ax1.plot(train_losses, "b-", linewidth=2)
    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Test accuracy
    ax2.plot(test_accuracies, "r-", linewidth=2)
    ax2.set_title("Test Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3.plot(learning_rates, "g-", linewidth=2)
    ax3.set_title("Learning Rate", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Combined plot
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(train_losses, "b-", linewidth=2, label="Train Loss")
    line2 = ax4_twin.plot(test_accuracies, "r-", linewidth=2, label="Test Accuracy")

    ax4.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss", color="blue")
    ax4_twin.set_ylabel("Accuracy", color="red")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="center right")

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("enhanced_training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def test_model_inference():
    """Test the trained model with some examples"""
    cfg = EnhancedConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load the best model
    if os.path.exists("enhanced_maple_best.pth"):
        checkpoint = torch.load("enhanced_maple_best.pth", map_location=device)

        model = EnhancedMaPLe(
            input_dim=28 * 28,
            num_classes=10,
            prompt_dim=cfg.prompt_dim,
            mask_ratio=cfg.mask_ratio,
            num_layers=cfg.num_layers,
            dropout_rate=cfg.dropout_rate,
            use_attention=cfg.use_attention,
            use_residual=cfg.use_residual,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Loaded model with best accuracy: {checkpoint['best_accuracy']:.4f}")
    else:
        print("No saved model found!")
        return

    # Test with some examples
    from data_manager import get_dataloaders

    _, test_loader = get_dataloaders(batch_size=8)

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images, training=False)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]

    print("\nInference Results:")
    print("Ground truth:", labels.tolist())
    print("Predicted:   ", predictions.cpu().tolist())
    print("Confidences: ", [f"{conf:.3f}" for conf in confidences.cpu().tolist()])

    # Visualize some results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")

        is_correct = predictions[i] == labels[i]
        color = "green" if is_correct else "red"
        ax.set_title(
            f"GT: {labels[i]} | Pred: {predictions[i].item()}\n"
            f"Conf: {confidences[i]:.3f}",
            color=color,
            fontsize=10,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("enhanced_inference_results.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Train the enhanced model
    model, best_acc = train_enhanced()

    # Test inference
    test_model_inference()
