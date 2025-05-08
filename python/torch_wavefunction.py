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




def train_wavefunction(model, x_train, epochs=1000, lr=1e-2, print_interval=100, save_wavefunction_history=False, previous_wavefunctions=None, overlap_penalty=1e2, callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dx = (x_train[1] - x_train[0]).real  # Real-valued step size

    energy_history = []
    wave_function_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        psi = model(x_train) # computes forward pass
        T = kinetic_energy(psi, x_train)
        V_psi = V(x_train) * psi
        H_psi = T + V_psi


        numerator = torch.trapezoid(torch.conj(psi) * H_psi, x_train, dim=0)
        denominator = torch.trapezoid(torch_abs2(psi), x_train, dim=0)
        energy = numerator / denominator

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

        loss.squeeze().backward()
        optimizer.step()

        energy_history.append(energy.item())
        
        if save_wavefunction_history and epoch%10==0 :
            norm=torch.sqrt(torch.sum(psi**2)*dx)
            psi_normalized=psi/norm
            wave_function_history.append(psi_normalized.detach().cpu().numpy())

        
        if epoch % print_interval == 0:
            print(f"Epoch {epoch}: Energy = {energy.item():.6f}")
        
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


if __name__ == "__main__":
    
    x_train = torch.linspace(-10, 10, 1000, dtype=torch.float32, device=device).view(-1, 1)
    x_train.requires_grad = True
    model = WaveFunctionMLP().to(device)
    early_stopping = EarlyStoppingCallback(patience=250, min_delta=1e-5)
    psi_normalized, energy_fin, energy_hist_0, wf_hist_0, psi_normalized_torch = train_wavefunction(model, x_train, epochs=1000, save_wavefunction_history=True, callback=early_stopping)
    model2 = WaveFunctionMLP([1,80,80,80,1]).to(device)
    early_stopping = EarlyStoppingCallback(patience=300, min_delta=1e-4)
    psi_normalized_2, energy_fin_2, energy_hist_1, wave_function_history,_ = train_wavefunction(model2, x_train, lr=0.001, previous_wavefunctions=[psi_normalized_torch], epochs=10000, callback=early_stopping, save_wavefunction_history=True,)
    def analytic_solution(x):
        return (1/np.pi)**0.25 * np.exp(-0.5*x**2)
    def np_abs2(x):
        """Compute the absolute square of a complex number."""
        return np.real(np.real(x)**2 + np.imag(x)**2)
    
    def torch_analytic_solution(x):
        return (1/np.pi)**0.25 * torch.exp(-0.5*x**2)
    
    with torch.no_grad():
        
        def exact_solution(x,n):
            hermite_poly = hermite(n)
            coeff = np.sqrt(1 / (2**n * factorial(n))) * (1 / np.pi)**0.25
            return coeff * np.exp(-0.5*x**2) * hermite_poly(x)
        
        def exact_energy(n):
            return n + 0.5
        
        import matplotlib.animation as animation
        
        def generate_anim(wave_function_history, x_train, n=0, filename="wavefunction_animation.gif"):
            fig, ax = plt.subplots()
            print(f"Wavefunction history size: {len(wave_function_history)}")
            psi_exact = exact_solution(x_train.cpu().numpy(), n)
            
            ax.plot(x_train.cpu().numpy(), np_abs2(psi_exact), label="Exact Solution", linestyle='--', color="#666666", linewidth=1)
            line, = ax.plot(x_train.cpu().numpy(), np_abs2(wave_function_history[0]), label="NQS", color="orange")
            ax.set_ylim(-0.1, 0.65)
            ax.set_ylabel("|ψ|²")
            ax.set_xlabel("x")
            ax.legend()
            def update(frame):
                line.set_ydata(np_abs2(wave_function_history[frame]))
                ax.set_title(f"Epoch {frame*10}")
                return line,
            plt.close(fig)  # Close the figure to prevent it from displaying immediately

            ani = animation.FuncAnimation(fig, update, frames=len(wave_function_history), interval=50, blit=False)
            ani.save(filename, writer='pillow', fps=10)
            
        generate_anim(wf_hist_0, x_train, n=0, filename="GS.gif")
        generate_anim(wave_function_history, x_train,n=1, filename="1st.gif")

        def generate_energy_plot(energy_hists):
            for i, energy_hist in enumerate(energy_hists):
                plt.plot(energy_hist, label=f"{i}")
                plt.hlines([exact_energy(i) for i in range(len(energy_hists))], color='red', linestyle='--', label="Exact Energies", xmin=0, xmax=len(energy_hist))
            plt.xlabel("Epoch")
            plt.ylabel("Energy")
            plt.title("Energy Convergence")
            plt.loglog()
            plt.legend()
            plt.show()
        generate_energy_plot([energy_hist_0, energy_hist_1])
        
        
        
        final_wavefunction = model(x_train)
        normalization = torch.sqrt(torch.trapz(torch_abs2(final_wavefunction), x_train,dim=0))
        
        final_wavefunction /= normalization
        final_wavefunction_cpu = final_wavefunction.cpu().numpy()
        
        probability_density = torch_abs2(final_wavefunction).cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.plot(x_train.cpu().numpy(), psi_normalized, label="Probability Density")
        plt.plot(x_train.cpu().numpy(), psi_normalized_2, label="Probability Density 2")
        plt.xlabel("x")
        plt.ylabel("Wavefunction")
        plt.title("Wavefunction History")
        plt.subplot(1,2,2)
        plt.plot(energy_hist_1, label="Energy History")
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.plot(energy_hist_0, label="Energy History 0")
        plt.title("Energy Convergence")
        
        plt.show()
    
        
# if __name__ == "__main__":
#     Nh = 10
#     params = {
#         "b": torch.randn(1, dtype=torch.float32, device=device),
#         "c": torch.randn(Nh, dtype=torch.float32, device=device),
#         "w": torch.randn(Nh, dtype=torch.float32, device=device),
#         "Nh": Nh,
#         "is_complex": False
#     }
#     model = WaveFunctionNN(params).to(device)

#     x_train = torch.linspace(-10, 10, 1000, dtype=torch.float32, device=device)
#     x_train.requires_grad = True

#     psi_normalized, energy_fin, energy_hist_0, _, psi_normalized_torch = train_wavefunction(model, x_train)
    
#     complex_type = torch.complex128
    
#     params = {
#         "b": torch.randn(1, dtype=complex_type, device=device),
#         "c": torch.randn(Nh, dtype=complex_type, device=device),
#         "w": torch.randn(Nh, dtype=complex_type, device=device),
#         "Nh": Nh,
#         "is_complex": True
#     }
#     model = WaveFunctionNN(params).to(device)

#     x_train = torch.linspace(-10, 10, 1000, dtype=torch.float32, device=device)
#     x_train.requires_grad = True
#     psi_normalized_2, energy_fin_2, energy_hist_1, wave_function_history,_ = train_wavefunction(model, x_train, previous_wavefunctions=[psi_normalized_torch], epochs=10000, lr=0.001)
    
#     def analytic_solution(x):
#         return (1/np.pi)**0.25 * np.exp(-0.5*x**2)
#     def np_abs2(x):
#         """Compute the absolute square of a complex number."""
#         return np.real(np.real(x)**2 + np.imag(x)**2)
    
#     def torch_analytic_solution(x):
#         return (1/np.pi)**0.25 * torch.exp(-0.5*x**2)
    
#     with torch.no_grad():
    
        
#         final_wavefunction = model(x_train)
#         normalization = torch.sqrt(torch.trapz(torch_abs2(final_wavefunction), x_train))
        
#         final_wavefunction /= normalization
#         final_wavefunction_cpu = final_wavefunction.cpu().numpy()
        
#         # is_fin_wf_normalized = torch.allclose(torch.trapz(torch_abs2(final_wavefunction), x_train), torch.tensor(1.0, device=device), atol=1e-6)
#         # print(f"Is final wavefunction normalized? {is_fin_wf_normalized}")
        
#         probability_density = torch_abs2(final_wavefunction).cpu().numpy()

#         plt.figure(figsize=(10, 5))
#         plt.subplot(1,2,1)
#         plt.plot(x_train.cpu().numpy(), psi_normalized, label="Probability Density")
#         plt.plot(x_train.cpu().numpy(), psi_normalized_2, label="Probability Density 2")
#         plt.xlabel("x")
#         plt.ylabel("Wavefunction")
#         plt.title("Wavefunction History")
#         plt.subplot(1,2,2)
#         plt.plot(energy_hist_1, label="Energy History")
#         plt.xlabel("Epoch")
#         plt.ylabel("Energy")
#         plt.plot(energy_hist_0, label="Energy History 0")
#         plt.title("Energy Convergence")
        
#         plt.show()
        
        