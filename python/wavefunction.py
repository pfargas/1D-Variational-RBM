import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.special import hermite
from scipy.optimize import dual_annealing


class Wavefunction:
    def __init__(self, L, b, c=None, w=None):
        """+
        Initialize the wavefunction parameters.
        Parameters:
        L (float): Parameter lambda: bias of the quadratic term
        b (float): Parameter b: bias of the linear term
        c (list): List of parameters c: bias of the hidden units
        w (list): List of parameters w: weights of the hidden units
        """

        self.L = L
        self.b = b
        self.c = c 
        self.w = w
        self.Nh = len(c) if c is not None else 0
        self.x = None
        self.psi_numeric = None
        self.normalization = None
        self.wavefunction: callable = None

    def _normalization(self):
        def integrand(x):
            return self._wavefunction(x)**2

        norm, _ = quad(integrand, -np.inf, np.inf)
        return 1 / np.sqrt(norm)
        
    def _wavefunction(self, x):
        prod = 1
        for i in range(self.Nh):
            prod *= 1+np.exp(self.c[i]+x*self.w[i])
        return np.exp(-self.L*x**2 + x*self.b)*prod

    def compute(self, x):
        self.x = x
        self.psi_numeric = self._wavefunction(x)
        self.normalization = self._normalization()
        self.psi_numeric *= self.normalization
        self.wavefunction = lambda x: self._wavefunction(x) * self.normalization
        
        return self.psi_numeric
    
    def analytic_solution(self, x, n=0):
        hermite_poly = hermite(n)
        coeff = np.sqrt(1 / (2**n * factorial(n))) * (1 / np.pi)**0.25
        return coeff * np.exp(-x**2 / 2) * hermite_poly(x)
    
    def is_normalized_analyt(self):
        norm = quad(lambda x: self.analytic_solution(x)**2, -np.inf, np.inf)[0]
        return np.isclose(norm, 1.0, atol=1e-10)

    def plot(self, analytic=True):
        if self.x is None or self.psi_numeric is None:
            raise ValueError("Wavefunction not computed. Call compute() first.")
        
        if analytic:
            plt.plot(self.x, self.analytic_solution(self.x)**2, label='Analytic Solution', color='red')
        
        plt.plot(self.x, self.wavefunction(self.x)**2, label='Variational Wavefunction', color='blue')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title('Wavefunction Probability Density')
        plt.show()
    
    def kinetic_energy(self):
        kin = -0.5 *self.psi_numeric * np.gradient(np.gradient(self.psi_numeric, self.x), self.x)
        return simpson(kin, self.x)
    
    def potential_energy(self):
        return 0.5*simpson((self.psi_numeric*self.x)**2, self.x)
    
    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()
    
    def local_energy(self):
        local_kin = -0.5 * np.gradient(np.gradient(self.psi_numeric, self.x), self.x)/self.psi_numeric
        local_kin = np.nan_to_num(local_kin, nan=0.0)
        local_pot = 0.5 * self.x**2
        return local_kin + local_pot
    
    def total_energy_from_local(self):
        local_energy = self.local_energy()
        return simpson(self.psi_numeric**2*local_energy, self.x)
        
    def an_potential_energy(self, n):
        return 0.5 * simpson((self.analytic_solution(self.x, n) * self.x)**2, self.x)
    
    def an_total_energy(self, n):
        return self.an_kin_energy(n) + self.an_potential_energy(n)
    

def variational_optimization(params):
    
    # b = params["b"]
    # c = params["c"]
    # w = params["w"]
    Nh = params["Nh"]
    
    def objective_function(x):
        b = x[0]
        c = x[1:1+Nh]
        w = x[1+Nh:]
        wavefunction = Wavefunction(0.5, b, c, w)
        x_space = np.linspace(-5, 5, 1000)
        wavefunction.compute(x_space)
        
        energy = wavefunction.total_energy()
        return energy

    bounds = [(0, 1)]
    for i in range(Nh):
        # Add bounds for c
        bounds.append((-0.1, 0.1)) # LOOK INTO THIS
    for i in range(Nh):
        # Add bounds for w
        bounds.append((-0.1, 0.1))
    return dual_annealing(objective_function, bounds)

        