import tqdm
import csv
import matplotlib.pyplot as plt

class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()

        data_loader = tqdm.tqdm(self.data_loader, desc='Training', unit='batch')
        for step, batch_data in enumerate(data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()

    def save_metrics_to_csv(self, epoch, loss, metric_value, filename):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, loss, metric_value])

    def plot_and_save_metrics(self, epochs, train_losses, val_losses, train_metrics, val_metrics):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('save/Train_results/loss_plot.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_metrics, label='Train Mean IoU')
        plt.plot(epochs, val_metrics, label='Validation Mean IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.title('Training and Validation Mean IoU')
        plt.legend()
        plt.grid(True)
        plt.savefig('save/Train_results/miou_plot.png')
        plt.close()