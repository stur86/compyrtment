import numpy as np
from collections import OrderedDict
from scipy.integrate import odeint


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

        self._D1 = np.zeros((self._N, self._N))
        self._D1m = self._D1.copy().astype(bool)    # Mask

        self._D2 = np.zeros((self._N, self._N, self._N))
        self._D2m = self._D2.copy().astype(bool)    # Mask

        self._couplings = OrderedDict()
        self._cdata = {
            'C': np.zeros(0),
            'i': np.zeros((0, 4)).astype(int)
        }

    @property
    def size(self):
        return self._N

    @property
    def states(self):
        return list(self._states)

    @property
    def couplings(self):
        return OrderedDict(self._couplings)

    def set_coupling_rate(self, descr, C=0, name=None):
        """Set a coupling between states

        Set a coupling between two states of the model. The coupling can be
        of three types:

        * CONSTANT
            A constant rate of growth for a certain state of the 
            model.
        * LINEAR
            A rate of growth that is linearly proportional to the population
            of the same or another state.
        * QUADRATIC
            A rate of growth that is proportional to the product of the 
            populations of two states.

        The coupling is defined by a descriptor string. This can have up to 
        four states:

        "S1*S2:S3=>S4"

        defines a quadratic coupling, controlled by the S1*S2 product, from S3
        to S4. The argument C controls the coupling. This means having:

        dS3/dt = -C*S1*S2

        dS4/dt = C*S1*S2

        If S3 and S4 are both missing, S3=S1 and S4=S2 is assumed. If one of
        S1 or S2 is missing, the coupling is taken to be linear, proportional
        to the one that's present. If both are missing, the growth is taken
        to be constant. Sign is controlled by the arrow; by convention, S4
        sees a term proportional to C, and S3 to -C. Either term can be
        missing, in which case we have a source or sink.
        Examples:

        "S1:=>S4"

        results in 

        dS4/dt = C*S1

        "S3=>S4"

        results in 

        dS3/dt = -C

        dS4/dt = C

        and so on. 

        Arguments:
            descr {str} -- Descriptor string for the coupling.

        Keyword Arguments:
            C {number} -- Coupling constant (default: {0})
            name {str} -- Name of coupling (default: {None})
        """

        # Parse the description
        if ':' in descr:
            s12, s34 = descr.split(':')
        elif '=>' in descr:
            s12 = None
            s34 = descr
        else:
            s12 = descr
            s34 = None

        if s12 is None or s12 == '':
            s1 = None
            s2 = None
        else:
            if '*' in s12:
                s1, s2 = s12.split('*')
                s1 = None if s1 == '' else s1
                s2 = None if s2 == '' else s2
                if s1 is None:
                    s1, s2 = s2, s1
            else:
                s1 = s12
                s2 = None

        if s34 is None or s34 == '':
            s3 = None
            s4 = None
        else:
            if '=>' in s34:
                s3, s4 = s34.split('=>')
                s3 = None if s3 == '' else s3
                s4 = None if s4 == '' else s4
            else:
                s3 = None
                s4 = s34

        if not all(s in self._states for s in (s1, s2, s3, s4)
                   if s is not None):
            raise(ValueError('Invalid state names used in coupling '
                             'definition'))

        # What kind of coupling is it?

        i1 = self._states.index(s1)
        i2 = self._states.index(s2) if s2 is not None else self._N
        i3 = self._states.index(s3) if s3 is not None else self._N
        i4 = self._states.index(s4) if s4 is not None else self._N

        if i3+i4 == 2*self._N:
            if i2 == self._N:
                i4 = i1
            else:
                i3 = i1
                i4 = i2

        if name is None:
            descr = "{0}*{1}:{2}=>{3}".format(*[s if s is not None else ''
                                                for s in (s1, s2, s3, s4)])
            name = descr

        self._couplings[name] = (descr, C)

        self._cdata['C'] = np.concatenate([self._cdata['C'], [C]])
        self._cdata['i'] = np.concatenate(
            [self._cdata['i'], [[i1, i2, i3, i4]]], axis=0)

    def dy_dt(self, y):
        """Time derivative from a given state

        Compute the time derivative of the model for a given state vector.

        Arguments:
            y {np.ndarray} -- State vector

        Returns:
            [np.ndarray] -- Time derivative of y
        """

        yext = np.concatenate([y, [1]])
        dydt = yext*0.

        C = self._cdata['C']
        i1, i2, i3, i4 = self._cdata['i'].T

        cy = C*yext[i1]*yext[i2]
        np.add.at(dydt, i3, -cy)
        np.add.at(dydt, i4,  cy)

        return dydt[:-1]

    def d2y_dtdC(self, y, dydC):
        """Time derivative of the model's gradient

        Time derivative of the gradient of the model's state with respect
        to all its parameters.

        Arguments:
            y {np.ndarray} -- State vector
            dydC {np.ndarray} -- Gradient of y w.r.t. coupling parameters
        """

        C = self._cdata['C']
        nC = len(C)
        i1, i2, i3, i4 = self._cdata['i'].T

        yext = np.concatenate([y, [1]])
        dydCext = dydC.reshape(-1, self._N)
        if dydCext.shape[0] != nC:
            raise ValueError('Invalid size gradient passed to d2y_dtdC')
        dydCext = np.concatenate(
            [dydCext, np.zeros((nC, 1))], axis=1)
        d2ydtdC = dydCext*0.

        dcy1 = yext[i1]*yext[i2]

        np.add.at(d2ydtdC, (range(nC), i3), -dcy1)
        np.add.at(d2ydtdC, (range(nC), i4),  dcy1)

        iC = np.repeat(np.arange(nC), nC)
        i1t = np.tile(i1, nC)
        i2t = np.tile(i2, nC)
        i3t = np.tile(i3, nC)
        i4t = np.tile(i4, nC)

        dcy2 = np.tile(C, nC)*(dydCext[iC, i1t]*yext[i2t] +
                               dydCext[iC, i2t]*yext[i1t])

        np.add.at(d2ydtdC, (iC, i3t), -dcy2)
        np.add.at(d2ydtdC, (iC, i4t),  dcy2)

        d2ydtdC = d2ydtdC[:, :-1].reshape(-1)

        return d2ydtdC

    def integrate(self, t, y0, use_gradient=False):

        if not use_gradient:
            def ode(y, t):
                return self.dy_dt(y)

            return odeint(ode, y0, t)

        else:
            def ode(y, t):
                dydt = y*0.
                dydt[:self._N] = self.dy_dt(y[:self._N])
                dydt[self._N:] = self.d2y_dtdC(y[:self._N], y[self._N:])

                return dydt

            nC = len(self._couplings)
            y0 = np.concatenate([y0, np.zeros(nC*self._N)])

            traj = odeint(ode, y0, t)

            ans = {'y': traj[:, :self._N]}

            for i, name in enumerate(self._couplings.keys()):
                ans['dy/d({0})'.format(name)] = traj[:,
                                                     (i+1)*self._N:(i+2) *
                                                     self._N]

            return ans

    @staticmethod
    def make_SIR(beta=0.3, gamma=0.2):

        sir = CModel('SIR')
        sir.set_coupling_rate('S*I', beta, name='beta')
        sir.set_coupling_rate('I:I=>R', gamma, name='gamma')

        return sir

    @staticmethod
    def make_LotkaVolterra(alpha=2.0/3.0, beta=4.0/3.0, gamma=1, delta=1):

        lv = CModel('Pp')
        lv.set_coupling_rate('p:=>p', alpha, 'alpha')
        lv.set_coupling_rate('p*P:p=>', beta, 'beta')
        lv.set_coupling_rate('p*P:=>P', gamma, 'gamma')
        lv.set_coupling_rate('P:P=>', delta, 'delta')

        return lv