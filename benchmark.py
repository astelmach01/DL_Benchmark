import psutil
import platform
import logging
import time

import torch
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
from tqdm import tqdm

BATCH_SIZE = 32


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)


# Function to get the device
def get_device() -> str:
    if torch.cuda.is_available():
        logging.info(f"Running CUDA on {torch.cuda.device_count()} devices")
        return "cuda"
    elif torch.backends.mps.is_available():
        logging.info("Running on MPS")
        return "mps"
    else:
        logging.info("Running on CPU")
        return "cpu"


# Function to load the model without pre-trained weights
def load_model(device):
    model = vit_b_16(weights=None)
    model.to(device)
    return model


# Function to prepare data
def prepare_data():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


# Function to calculate F1 score
def calculate_f1(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    labels_cpu = labels.cpu().numpy()
    predicted_cpu = predicted.cpu().numpy()
    return f1_score(labels_cpu, predicted_cpu, average="weighted")


# Function for training
def train_model(model, data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1  # Adjust as needed

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_accuracy, running_f1 = 0.0, 0.0, 0.0
        with tqdm(data_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += calculate_accuracy(outputs, labels)
                running_f1 += calculate_f1(outputs, labels)

                tepoch.set_postfix(
                    loss=running_loss / len(data_loader),
                    accuracy=running_accuracy / len(data_loader),
                    f1=running_f1 / len(data_loader),
                )

    end_time = time.time()
    return end_time - start_time


def main():
    device_type = get_device()
    device = torch.device(device_type)
    logging.info(f"Using device: {device_type}")

    # Setting seeds for reproducibility
    torch.manual_seed(0)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(0)

    model = load_model(device)
    data_loader = prepare_data()
    training_time = train_model(model, data_loader, device)
    print(f"Training Time: {training_time} seconds")
    
    # Get system info
    device = platform.system()
    processor = platform.processor()
    ram = str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    storage = str(round(psutil.disk_usage('/').total / (1024.0 **3))) + " GB SSD"

    # Print stats
    print("Device\tProcessor\tRAM\tStorage\tBatch Size\tTime (h:m:s)")
    print(f"{device}\t{processor}\t{ram}\t{storage}\t{BATCH_SIZE}\t{training_time}")


if __name__ == "__main__":
    main()
