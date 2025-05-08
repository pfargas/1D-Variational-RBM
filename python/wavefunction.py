import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.special import hermite
from scipy.optimize import dual_annealing, basinhopping
import torch

def np_abs2(x):
    """Compute the absolute square of a complex number."""
    return np.real(np.real(x)**2 + np.imag(x)**2)


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
            return np_abs2(self._wavefunction(x))

        norm, _ = quad(integrand, -np.inf, np.inf)
        return 1 / np.sqrt(norm)
        
    def _wavefunction_old(self, x):
        prod = 1
        for i in range(self.Nh):
            prod *= 1+np.exp(self.c[i]+x*self.w[i])
        return np.exp(-self.L*x**2 + x*self.b)*prod
    
    def _wavefunction(self, x):
        add = 0
        for i in range(self.Nh):
            add += np.log(1+np.exp(self.c[i]+x*self.w[i]))
        return np.exp(-self.L*x**2 + x*self.b + add)

    def compute(self, x):
        self.x = x
        self.psi_numeric = self._wavefunction(x)
        self.normalization = self._normalization()
        self.psi_numeric *= self.normalization
        self.wavefunction = lambda x: self._wavefunction(x) * self.normalization
        return self.psi_numeric
    
    @property
    def probability_density(self):
        if self.psi_numeric is None:
            raise ValueError("Wavefunction not computed. Call compute() first.")
        return np_abs2(self.psi_numeric)
    @property
    def probability_density_analytic(self):
        if self.x is None:
            raise ValueError("Wavefunction not computed. Call compute() first.")
        return np_abs2(self.analytic_solution(self.x))
    
    def analytic_solution(self, x, n=0):
        hermite_poly = hermite(n)
        coeff = np.sqrt(1 / (2**n * factorial(n))) * (1 / np.pi)**0.25
        return coeff * np.exp(-x**2 / 2) * hermite_poly(x)
    
    def is_normalized_analyt(self):
        norm = quad(lambda x: np_abs2(self.analytic_solution(x)), -np.inf, np.inf)[0]
        return np.isclose(norm, 1.0, atol=1e-10)

    def plot(self, analytic=True):
        if self.x is None or self.psi_numeric is None:
            raise ValueError("Wavefunction not computed. Call compute() first.")
        
        if analytic:
            plt.plot(self.x, self.probability_density_analytic, label='Analytic Solution', color='red')
        
        plt.plot(self.x, self.probability_density, label='Variational Wavefunction', color='blue')
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
        return np.real(self.kinetic_energy() + self.potential_energy())
    
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
    
    def GS_overlap(self):
        """
        ..math::
            \langle \phi_0 | \psi \rangle = \int_{-\infty}^{\infty} \phi_0(x) \psi(x) dx
        where :math:`\phi_0` is the ground state wavefunction and :math:`\psi` is the variational wavefunction.
        """
        def integrand(x):
            return self.analytic_solution(x) * self.wavefunction(x)
        
        def real_integrand(x):
            return self.analytic_solution(x) * (np.real(self.wavefunction(x)))
        
        def imag_integrand(x):
            return self.analytic_solution(x) * (np.imag(self.wavefunction(x)))
        
        real_overlap, _ = quad(real_integrand, -np.inf, np.inf)
        imag_overlap, _ = quad(imag_integrand, -np.inf, np.inf)
        overlap = real_overlap + 1j*imag_overlap
        return np_abs2(overlap)
    

def variational_optimization(params):
    
    # b = params["b"]
    # c = params["c"]
    # w = params["w"]
    Nh = params["Nh"]
    if "c_re" in params:
        is_complex = True
    else:
        is_complex = False
    
    dual_annealing = params.get("dual_annealing", False)
    
    def objective_function(x):
        b = x[0]
        if is_complex:
            c_re = x[1:1+Nh]
            c_im = x[1+Nh:1+2*Nh]
            c = c_re + 1j*c_im
            w_re = x[1+2*Nh:1+3*Nh]
            w_im = x[1+3*Nh:1+4*Nh]
            w = w_re + 1j*w_im
            if len(x)==Nh*4+2:
                lagrange = x[-1]
            else:
                lagrange = 1e2
        else:
            c = x[1:1+Nh]   
            w = x[1+Nh:Nh*2+1]
            if len(x)==Nh*2+2:
                lagrange = x[-1]
            else:
                lagrange = 1e2
        wavefunction = Wavefunction(0.5, b, c, w)
        x_space = np.linspace(-5, 5, 1000)
        wavefunction.compute(x_space)
        
        energy = wavefunction.total_energy()
        return energy + lagrange * wavefunction.GS_overlap()

    bounds = [(0, 1)]
    for i in range(Nh):
        # Add bounds for c
        bounds.append((-0.1, 0.1)) # real part
        if "c_re" in params:
            bounds.append((-0.1, 0.1)) # imaginary part
    for i in range(Nh):
        # Add bounds for w
        bounds.append((-0.1, 0.1))
        if "w_re" in params:
            bounds.append((-0.1, 0.1))
    if "lagrange" in params:
        bounds.append((0, 1e3))

    if dual_annealing:
        return dual_annealing(objective_function, bounds)
    else:
        if "c_re" not in params:
            params["initial"] = np.concatenate(([params["b"]], params["c"], params["w"]))
        else:
            params["initial"] = np.concatenate(([params["b"]], params["c_re"], params["c_im"], params["w_re"], params["w_im"]))
        if "lagrange" in params:
            params["initial"] = np.concatenate((params["initial"], [params["lagrange"]]))
        print(params["initial"])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        result = basinhopping(objective_function, params["initial"], niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=minimizer_kwargs)
        return result

        