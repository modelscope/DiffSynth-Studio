from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    batch_size = 64
    data_size = 10000
    num_epochs = 100
    dtype = torch.bfloat16

    input_dim = 10
    hidden_dim = 256
    output_dim = 2

    # generate random data
    x = torch.randn(data_size, input_dim)
    y = torch.randn(data_size, output_dim)
    
    # create dataset and dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # create model
    model = SimpleNet(input_dim, hidden_dim, output_dim)
    
    # configure deepspeed plugin
    deepspeed = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)

    # accelerator
    accelerator = Accelerator(deepspeed_plugin=deepspeed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    
    # prepare model and optimizer with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # train
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader)
        for step, datas in enumerate(pbar):
            inputs, labels = datas
            inputs = inputs.to(accelerator.device, dtype=dtype)
            labels = labels.to(accelerator.device, dtype=dtype)

            with accelerator.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    
       

