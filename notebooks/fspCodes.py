import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from pyexpokit import expmv

def solveFSP(A, P0, t_eval, rtol=1e-4, atol=1e-6):
    """ Sparse FSP system solver, dP/dt = A @ P, using expokit
    Input:
        A: sparse matrix
        P0: initial condition
        t_eval: time points to evaluate the solution
        rtol: relative tolerance
        atol: absolute tolerance
    """
    solution = np.zeros((len(t_eval), P0.shape[0]))
    P = P0
    solution[0,:] = P
    for i in range(len(t_eval)-1):
        deltaT = t_eval[i+1] - t_eval[i]
        P = expmv(deltaT, A, P)
        solution[i+1,:] = P
    return solution
    

def buildFSP(S, W, states, pars, maxSpecies, t=0):
    """ Build the infinitesimal generator matrix for a continuous time Markov chain using the
    compressed sparse row (CSR) format.
    Input:
        S: Stoichiometry matrix
        W: Propensity vector function
        states: list of states in FSP approximation
        pars: reaction rate parameters
        maxSpecies: maximum number of species in the FSP approximation
        t: time (for time inhomogeneous models)
    """
    
    # Determine the number of species, states and reactions
    nSpecies, nStates = states.shape
    nReactions = S.shape[1]
    
    # Compute the propensity functions for all states.
    propensities = W(states, t, pars)

    # Create the infinitesimal generator matrix as sparse matrix
    infGen = sp.csr_matrix((nStates+1, nStates+1))
    orig_indices = np.arange(nStates)

    vec = np.ones(nSpecies, dtype=int)  # Vector to transform the state to the index
    for i in range(nSpecies):
        vec[i] = np.prod(maxSpecies[i+1:,0]+1)  # The number of states for the next species

    # Step through each reaction
    for mu in range(nReactions):
        # Create a sparse matrix for the current reaction
        # Outgoing reactions (diagnoal elements of the sparse matrix)
        values = propensities[mu,:]

        # Incoming reactions (off-diagonal elements of the sparse matrix)
        # add the NxM states matrix to tthe Nx1 column vector
        newStates = states + S[:,mu][:,None].dot(np.ones((1,nStates)))

        # Check which states leave the system
        notSink = (newStates<=maxSpecies)*(newStates>=0)    
        notSink = np.all(notSink, axis=0)
        
        new_indices = nStates*np.ones((nStates), dtype=int)
        new_indices[notSink] = vec@(newStates[:,notSink])

        # Create the sparse matrix
        row_indices = np.concatenate((orig_indices, new_indices))
        col_indices = np.concatenate((orig_indices, orig_indices))
        values = np.concatenate((-1.*values, values))
        
        # Add the current reaction to the infinitesimal generator
        infGen += sp.csr_matrix((values, (row_indices, col_indices)), shape=(nStates+1, nStates+1))
    
    return infGen
