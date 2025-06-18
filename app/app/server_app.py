"""app: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from app.task import Net, get_weights, set_weights, test
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize

# Load global test set from local files
data_dir = Path("../Data")  # Adjust if needed
test_dir = data_dir / "test"
test_transforms = Compose(
    [
        Grayscale(num_output_channels=1),
        Resize((376, 480)),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)
testset = ImageFolder(str(test_dir), transform=test_transforms)
testloader = DataLoader(testset, batch_size=32)


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""
        # Instantiate model
        net = Net()
        # Apply global_model parameters
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Run test
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / total_examples}


def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.001
    # Appply a simple learning rate decay
    if server_round > 2:
        lr = 0.0005
    return {"lr": lr}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
