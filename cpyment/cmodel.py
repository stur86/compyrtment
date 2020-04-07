import numpy as np


class CModel(object):
    """CModel

    A compartment model. Has a number of states and evolves in time with three
    types of terms:

    * Constant rate
                
                d y_i / d_t = D^(0)_i

    * Linear rate
                
                d y_i / d_t = D^(1)_ij y_j

    * Quadratic rate
                
                d y_i / d_t = D^(2)_ijk y_j y_k

    """

    def __init__(self, states=''):
        """Initialise a CModel
        
        Initialise a compartment model to have the given states.
        
        Keyword Arguments:
            states {str or list} -- States of the model. Can be a string,
                                    in which case each state will be named by
                                    one letter, or a list of strings. State
                                    names must be distinct (default: {''})
        """ 

        # Initialise states
        self._N = len(states)
        self._states = list(states)

        if len(set(states)) != self._N:
            raise ValueError('State list has repeated entries')

        # Now, the couplings
        self._D0 = np.zeros((self._N))
        self._D0m = self._D0.copy().astype(bool)    # Mask

        self._D1 = np.zeros((self.N, self.N))
        self._D1m = self._D1.copy().astype(bool)    # Mask

        self._D2 = np.zeros((self.N, self.N, self._N))
        self._D2m = self._D2.copy().astype(bool)    # Mask

    @property
    def size(self):
        return self._N

    @property
    def states(self):
        return list(self._states)

    @property
    def D0(self):
        return self._D0.copy()
    
    @property
    def D1(self):
        return self._D1.copy()
    
    @property
    def D2(self):
        return self._D2.copy()
    
    
    
    
