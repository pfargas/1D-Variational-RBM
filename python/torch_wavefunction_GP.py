import torch
import torch.nn as nn
import numpy as np
from math import factorial
import copy
import matplotlib.pyplot as plt
from scipy.special import hermite


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def kinetic_energy(psi, x):
    # First derivative
    dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                  create_graph=True, retain_graph=True)[0]
    # Second derivative
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx),
                                    create_graph=True, retain_graph=True)[0]
    return -0.5 * d2psi_dx2.real

def V(x):
    return 0.5 * x**2

def V_GP(x, psi, g=0.000433*1000000):
    # Compute the potential energy term
    V_psi = 0.5 * g * psi * torch.conj(psi)
    return V_psi


class WaveFunctionNN(nn.Module):

    def __init__(self, params):
        super(WaveFunctionNN, self).__init__()
        self.type_var = params.get("type", torch.complex128)
        self.b = nn.Parameter(torch.tensor(params["b"], dtype=self.type_var, device=device))
        self.c = nn.Parameter(torch.tensor(params["c"], dtype=self.type_var, device=device))
        self.Nh = params["Nh"]
        self.w = nn.Parameter(torch.tensor(params["w"], dtype=self.type_var, device=device))
        self.is_complex = params.get("is_complex", False)
        
    def forward(self, x):
        add = 0
        
        # add = torch.sum(torch.log1p(torch.exp(self.c + x * self.w)), dim=1)
        
        for i in range(self.Nh//2):
            add += torch.log1p(torch.exp(self.c[i]+x*self.w[i]))
        return torch.exp(-0.5*x**2 + x*self.b + add)
    
def torch_abs2(x):
    """Compute the absolute square of a complex number."""
    try:
        result =  torch.real(torch.real(x)**2 + torch.imag(x)**2)
    except RuntimeError:
        result = torch.real(x**2)
    return result

class WaveFunctionRBM_OHE(nn.Module):
    
    def __init__(self, Nv = 50, Nh = 10):
        super(WaveFunctionRBM_OHE, self).__init__()
        self.b = nn.Parameter(torch.randn(Nv, dtype=torch.float32, device=device))
        self.c = nn.Parameter(torch.randn(Nh, dtype=torch.float32, device=device))
        self.w = nn.Parameter(torch.randn(Nv,Nh, dtype=torch.float32, device=device))
        self.Nh = Nh
        self.Nv = Nv
        self.xmin = -10
        self.xmax = 10
        self.dx = (self.xmax - self.xmin) / (self.Nv - 1)

    def forward(self, x):
        # Compute indices for one-hot encoding
        indices = ((x - self.xmin) / self.dx).long()  # [batch_size] tensor of ints

        # Clamp indices to be within valid range
        indices = torch.clamp(indices, 0, self.Nv - 1)
        # batch size is basically the number of points in the interval of computation
        # Nv is the discretization of the interval
        # Create one-hot encoded input: shape [batch_size, Nv]
        v = torch.zeros((x.shape[0], self.Nv), dtype=torch.float32, device=x.device)
        v.scatter_(1, indices.unsqueeze(1), 1.0)

        # Compute product over hidden units
        exponent = self.c + v @ self.w  # shape [batch_size, Nh]
        prod = torch.prod(1 + torch.exp(exponent), dim=1)  # shape [batch_size]

        # Final output
        output = torch.exp(v @ self.b) * prod  # shape [batch_size]
        return output

    # def forward(self, x):
    #     # one-hot encoding
    #     v = torch.zeros(self.Nv, dtype=torch.int8, device=device)
    #     v[int((x - self.xmin) / self.dx)] = 1
    #     prod = 1
    #     for i in range(self.Nh):
    #         prod *= 1+torch.exp(self.c[i] + v @ self.w[:, i])
    #     return torch.exp(v @ self.b)*prod
class WaveFunctionMLP(nn.Module):
    
    def __init__(self, layer_dims=[1,60,60,1]):
        super(WaveFunctionMLP, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # Adding activation function for all layers except the last lasyer
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.net(x)
    
class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_energy = None
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.stop_training = False

    def __call__(self, epoch, energy, model):
        # Initialize the best energy if it's the first epoch
        if self.best_energy is None:
            self.best_energy = energy
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
        else:
            # Check if the energy has improved by more than min_delta
            if energy < self.best_energy - self.min_delta:
                self.best_energy = energy
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        # If there is no improvement for the specified patience, stop training
        if self.epochs_without_improvement >= self.patience:
            print(f"Stopping training after {epoch+1} epochs due to no improvement in energy.")
            self.stop_training = True




def train_wavefunction(model, 
                       x_train, 
                       epochs=1000, 
                       lr=1e-2, 
                       print_interval=100, 
                       save_wavefunction_history=False, 
                       previous_wavefunctions=None, 
                       boundary_conditions=False,
                       overlap_penalty=1e2, 
                       callback=None, 
                       boundary_conditions_penalty=1e2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dx = (x_train[1] - x_train[0]).real  # Real-valued step size

    energy_history = []

    if save_wavefunction_history:
        wave_function_history = []
        psi = model(x_train)
        norm=torch.sqrt(torch.sum(psi**2)*dx)
        psi_normalized=psi/norm
        wave_function_history.append(psi_normalized.detach().cpu().numpy())
    else:
        wave_function_history = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        psi = model(x_train) # computes forward pass
        T = kinetic_energy(psi, x_train)
        V_psi = V(x_train) * psi
        V_GP_psi = V_GP(x_train, psi) * psi
        H_psi = T + V_psi + V_GP_psi


        numerator = torch.trapezoid(torch.conj(psi) * H_psi, x_train, dim=0)
        denominator = torch.trapezoid(torch_abs2(psi), x_train, dim=0)
        energy = numerator / denominator
        energy = energy.real

        loss = energy.real
        if previous_wavefunctions:
            orthogonality_loss = 0
            norm=torch.sqrt(torch.sum(psi**2)*dx)
            psi_normalized=psi/norm
            for prev_psi in previous_wavefunctions:
                overlap = torch.sum(psi_normalized * prev_psi)*dx
                orthogonality_loss += overlap**2  # Penalize the square of the overlap
            overlap_loss=overlap_penalty * orthogonality_loss
            loss +=overlap_loss.real  # Weight the orthogonality loss
        
        if boundary_conditions:
            norm=torch.sqrt(torch.sum(psi**2)*dx)
            psi_normalized=psi/norm
            boundary_loss = torch.sum(psi_normalized[0]**2) + torch.sum(psi_normalized[-1]**2)
            loss += boundary_conditions_penalty*boundary_loss

        loss.squeeze().backward()
        optimizer.step()

        energy_history.append(energy.item().real)
        
        if save_wavefunction_history and epoch%10==0 :
            norm=torch.sqrt(torch.sum(psi**2)*dx)
            psi_normalized=psi/norm
            wave_function_history.append(psi_normalized.detach().cpu().numpy())

        
        if epoch % print_interval == 0:
            print(f"Epoch {epoch}: Energy = {energy.item().real:.6f}")
        
        if callback is not None:
            callback(epoch, energy.item(), model)

            # If the callback indicates stopping, break the training loop
            if hasattr(callback, 'stop_training') and callback.stop_training:
              model.load_state_dict(callback.best_model_state)
              print(f"Training stopped early at epoch {epoch+1}")
              break

            
    with torch.no_grad():
        psi = model(x_train)
        psi_cpu = psi.cpu().numpy()
        energy = energy.item()
        # Normalize the wavefunction
        normalization = torch.sqrt(torch.trapz(torch_abs2(psi), x_train, dim=0))
        normalization_cpu = normalization.cpu().numpy()
        psi_normalized = psi_cpu / normalization_cpu
        psi_torch_normalized = psi / normalization


    return psi_normalized, energy,energy_history,wave_function_history,psi_torch_normalized
