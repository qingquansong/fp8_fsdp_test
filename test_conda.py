import torch
from lightning import LightningModule, Trainer

class SimpleModel(LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Create larger random data
x = torch.randn(64000, 10)  # Increase the number of samples
y = torch.randn(64000, 1)


# Create DataLoader
train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=32)

# Initialize the model
model = SimpleModel()

# Check if CUDA is availableprint(f"CUDA available: {torch.cuda.is_available()}")

# Run a simple training loop with Lightning
trainer = Trainer(max_epochs=100)
trainer.fit(model, train_loader)

