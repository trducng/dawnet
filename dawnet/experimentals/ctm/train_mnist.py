import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output, display
from torchvision import datasets, transforms



def get_loss(predictions, certainties, targets, use_most_certain=True):
    """use_most_certain will select either the most certain point or the final point."""

    losses = nn.CrossEntropyLoss(reduction="none")(
        predictions,
        torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1),
    )

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2
    return loss, loss_index_2


def calculate_accuracy(predictions, targets, where_most_certain):
    """Calculate the accuracy based on the prediction at the most certain internal tick."""
    B = predictions.size(0)
    device = predictions.device

    predictions_at_most_certain_internal_tick = (
        predictions.argmax(1)[torch.arange(B, device=device), where_most_certain]
        .detach()
        .cpu()
        .numpy()
    )
    accuracy = (
        targets.detach().cpu().numpy() == predictions_at_most_certain_internal_tick
    ).mean()

    return accuracy


def update_training_curve_plot(
    fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies, steps
):
    clear_output(wait=True)
    # Plot loss
    ax1.clear()
    ax1.plot(
        range(len(train_losses)),
        train_losses,
        "b-",
        alpha=0.7,
        label=f"Train Loss: {train_losses[-1]:.3f}",
    )
    ax1.plot(
        steps, test_losses, "r-", marker="o", label=f"Test Loss: {test_losses[-1]:.3f}"
    )
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.clear()
    ax2.plot(
        range(len(train_accuracies)),
        train_accuracies,
        "b-",
        alpha=0.7,
        label=f"Train Accuracy: {train_accuracies[-1]:.3f}",
    )
    ax2.plot(
        steps,
        test_accuracies,
        "r-",
        marker="o",
        label=f"Test Accuracy: {test_accuracies[-1]:.3f}",
    )
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    display(fig)


def train(model, trainloader, testloader, iterations, test_every, device):

    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=0.0001, eps=1e-8)

    model.train()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    steps = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    with tqdm(total=iterations, initial=0, dynamic_ncols=True) as pbar:
        test_loss = None
        test_accuracy = None
        for stepi in range(iterations):

            inputs, targets = next(iter(trainloader))
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, certainties, _ = model(inputs, track=False)
            train_loss, where_most_certain = get_loss(predictions, certainties, targets)
            train_accuracy = calculate_accuracy(
                predictions, targets, where_most_certain
            )

            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if stepi % test_every == 0 or stepi == iterations - 1:
                model.eval()
                with torch.inference_mode():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in testloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        predictions, certainties, _ = model(inputs, track=False)
                        test_loss, where_most_certain = get_loss(
                            predictions, certainties, targets
                        )
                        all_test_losses.append(test_loss.item())

                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(
                        all_test_where_most_certain, dim=0
                    )

                    test_accuracy = calculate_accuracy(
                        all_test_predictions,
                        all_test_targets,
                        all_test_where_most_certain,
                    )
                    test_loss = sum(all_test_losses) / len(all_test_losses)

                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)
                    steps.append(stepi)
                model.train()

                update_training_curve_plot(
                    fig,
                    ax1,
                    ax2,
                    train_losses,
                    test_losses,
                    train_accuracies,
                    test_accuracies,
                    steps,
                )

            pbar.set_description(
                f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}"
            )
            pbar.update(1)

    plt.ioff()
    plt.close(fig)
    return model


def prepare_data():
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root="/Users/john/dawnet/temp/data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="/Users/john/dawnet/temp/data", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=1, drop_last=False)
    return trainloader, testloader



if __name__ == "__main__":
    trainloader, testloader = prepare_data()
