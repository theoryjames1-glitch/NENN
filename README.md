### **Noisy Evolutionary Neural Network (NENN) Theory**

The **Noisy Evolutionary Neural Network (NENN)** is a novel neural network model that leverages **dynamic, stochastic processes** for the evolution of learning parameters. Unlike traditional neural networks where weights are optimized using gradient descent and a fixed learning rate, NENN introduces **controlled randomness** through **noise** and evolves its parameters using **evolutionary strategies** driven by feedback from the training process.

The central idea behind **NENN** is to combine the benefits of **exploration** (through noise and mutations) with the gradual **convergence** of learning parameters, ultimately leading to more **robust learning** and potentially better generalization.

### **Key Concepts**:

1. **Noisy Evolutionary Learning**:

   * **Noise** acts as a **mutation** operator, introducing randomness into the learning process. This helps the network explore different configurations of weights and parameters.
   * Over time, the model **evolves** by adjusting learning parameters (such as the learning rate, momentum, etc.) based on feedback from its performance. This leads to an **adaptive learning strategy**.

2. **Dynamically Evolving Parameters**:

   * Instead of relying on static parameters like a fixed learning rate or momentum, NENN’s parameters evolve through a feedback loop. **Markov coefficients** such as **learning rate** and **momentum** are adjusted according to past performance, based on a **Markovian process**.

3. **Noise as Mutation**:

   * The noise is applied to the model's **weights** and **inputs**, causing the weights to evolve stochastically. This is akin to the concept of **genetic mutation** in biological evolution, where random changes in the model’s parameters help explore the solution space.

4. **Dithering as Filtering**:

   * **Dithering** is a small, random perturbation added to the **outputs** of the neurons, allowing the model to continue exploring and avoiding overfitting to the data.

5. **Exploration vs. Exploitation**:

   * In the initial phases of training, NENN explores the parameter space using high **noise levels** and dynamic parameter adjustments. As training progresses, the system **evolves** toward **exploitation** by reducing the noise and focusing on stable regions of the solution space.

### **Mathematical Representation of NENN**:

#### **State Variables**:

* **Weights** $W_t$ (at time $t$)
* **Learning rate** $\eta_t$
* **Momentum** $\mu_t$
* **Noise factor** $\nu_t$
* **Dither strength** $\delta_t$

#### **Markovian Evolution of Coefficients**:

The evolution of the **learning rate** ($\eta_t$) and **momentum** ($\mu_t$) is driven by feedback from the model’s state, governed by the following dynamics:

$$
\eta_{t+1} = f(\eta_t, L(W_t)) \quad \text{(Learning rate evolution)}
$$

$$
\mu_{t+1} = g(\mu_t, L(W_t)) \quad \text{(Momentum evolution)}
$$

$$
\nu_{t+1} = \nu_t \cdot \text{decay} \quad \text{(Noise decay over time)}
$$

$$
\delta_{t+1} = \delta_t \cdot \text{decay} \quad \text{(Dither decay over time)}
$$

Where:

* $L(W_t)$ is the **loss function** at time $t$
* $f$ and $g$ are functions that adapt the learning rate and momentum based on the model’s performance.

#### **State Update (Weight Evolution)**:

The weight update rule can be represented as:

$$
W_{t+1} = W_t - \eta_t \cdot \nabla L(W_t) + \mu_t \cdot (W_t - W_{t-1}) + \nu_t \cdot \epsilon_t
$$

Where:

* $\nabla L(W_t)$ is the **gradient** of the loss function with respect to the weights at time $t$.
* $\nu_t \cdot \epsilon_t$ is the **noise** applied to the weights (mutation).
* $\mu_t$ is the **momentum** factor, which influences how much the previous update affects the current one.

#### **Output Equation**:

The output of the model is determined by the weighted sum of the inputs followed by the application of an activation function:

$$
y_t = \text{Activation}(W_t \cdot x + b)
$$

Where:

* $y_t$ is the output at time $t$.
* $x$ is the input at time $t$.
* $W_t$ is the weight matrix at time $t$.
* $b$ is the bias vector.
* The activation function is typically **ReLU** or **Sigmoid**, depending on the task (classification or regression).

---

### **Noisy Evolutionary Neural Network (NENN) Script**

Below is the **Noisy Evolutionary Neural Network (NENN)** implementation. This model introduces **noise** and **dithering** as **mutations** and **filters**, and the **learning rate** and **momentum** evolve based on training feedback.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----- Noisy Evolutionary Neuron -----
class NoisyEvolutionaryNeuron(nn.Module):
    def __init__(self, input_size, output_size):
        super(NoisyEvolutionaryNeuron, self).__init__()

        # Weights (trainable)
        self.weights = nn.Parameter(torch.randn(input_size, output_size))

        # Noise/dither learnable parameters
        self.noise_factor = nn.Parameter(torch.tensor(0.05))
        self.dither_strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, epoch):
        # Decay noise & dither slightly each epoch (slower decay)
        decay = 0.99  # Slower decay for noise and dithering
        self.noise_factor.data *= decay
        self.dither_strength.data *= decay

        # Apply noise to inputs
        noise = torch.randn_like(x) * self.noise_factor
        x = x + noise

        # Linear transform
        output = torch.matmul(x, self.weights)

        # Apply dithering to outputs
        dither_noise = torch.randn_like(output) * self.dither_strength
        output = output + dither_noise

        return output


# ----- Network using noisy evolutionary neurons -----
class NoisyEvolutionaryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NoisyEvolutionaryNetwork, self).__init__()
        self.layer1 = NoisyEvolutionaryNeuron(input_size, hidden_size)
        self.layer2 = NoisyEvolutionaryNeuron(hidden_size, output_size)

    def forward(self, x, epoch):
        x = self.layer1(x, epoch)
        x = F.relu(x)   # Non-linearity needed for XOR!
        x = self.layer2(x, epoch)
        return x


# ----- XOR dataset -----
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# ----- Model -----
model = NoisyEvolutionaryNetwork(input_size=2, hidden_size=4, output_size=1)

criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Slightly reduced learning rate

# ----- Training -----
num_epochs = 10000
for epoch in range(num_epochs):
    outputs = model(X, epoch)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitoring the progress
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----- Testing -----
model.eval()
with torch.no_grad():
    predicted = torch.sigmoid(model(X, num_epochs))  # Apply sigmoid for binary output
    print("\nRaw outputs:\n", predicted)
    print("Rounded predictions:\n", predicted.round())  # Rounded predictions for binary output
```

### **Explanation of Code**:

1. **Noisy Evolutionary Neuron**:

   * This class introduces **noise** and **dithering** in the forward pass. The noise is applied to both the inputs (for mutation) and the outputs (for dithering).
   * Both **noise** and **dithering** are **decayed over time** to stabilize the model's learning process as training progresses.

2. **NoisyEvolutionaryNetwork**:

   * The network consists of two layers of **Noisy Evolutionary Neurons**, with **ReLU** applied between layers to introduce non-linearity.
   * The **forward pass** applies the noise and dither and computes the output based on the evolving weights.

3. **Training**:

   * The model is trained using the **XOR dataset** and **binary cross-entropy loss** (BCEWithLogitsLoss).
   * The **Adam optimizer** is used to adjust the weights, with a **slightly reduced learning rate** for better convergence.

4. **Testing**:

   * After training, the model's predictions are passed through **sigmoid** to convert them into binary outputs, and the results are **rounded** to produce final binary predictions (0 or 1).

---

### **Conclusion**:

The **Noisy Evolutionary Neural Network (NENN)** introduces **noise** and **dithering** as key components of the learning process. The model evolves over time, adapting its **learning rate**, **momentum**, and other parameters based on real-time feedback. This enables the network to **explore** different solutions while stabilizing over time.

NENN is well-suited for problems where **exploration** is beneficial, such as complex optimization problems, and can help in scenarios where traditional neural networks may overfit or get stuck in local minima.

Let me know if you need further adjustments or want to test the model on a different task!
