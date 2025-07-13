import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.arch = nn.Sequential(
             nn.Linear(1, n),
             nn.ReLU(),
             nn.Linear(n, 1),
        )

    def forward(self, x):
        return self.arch(x)
    

def target_function(x):
    return torch.sigmoid(x)

def main():
    print("Please enter a function that you wish to approximate: ")

    # Extract function
    pass

    # Train
    x_train = torch.linspace(-5, 5, 200).reshape(-1, 1)
    y_train = target_function(x_train)

    model = NeuralNetwork(n=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for _ in range(100):
        model.train()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        W1 = model.arch[0].weight.detach()
    b1 = model.arch[0].bias.detach() 
    W2 = model.arch[2].weight.detach()
    b2 = model.arch[2].bias.detach()

    hidden_input = x_train @ W1.T + b1
    hidden_output = torch.relu(hidden_input)

    components = []
    for i in range(W1.shape[0]):
        contribution = hidden_output[:, i] * W2[0, i]
        components.append(contribution)

    x_vals = x_train.squeeze()

    plt.figure(figsize=(10, 6))
    for i, c in enumerate(components[:10]):
        plt.plot(x_vals, c, label=f"Neuron {i}", linewidth="1")

    total = torch.stack(components, dim=0).sum(dim=0)
    plt.plot(x_vals, total + b2.item(), label="NN Output", color="blue", linewidth=4)
    plt.plot(x_vals, y_train.squeeze(), label="Target", linestyle="--", color="red", linewidth=2)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Neuron Contributions (ReLU Basis Functions)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()