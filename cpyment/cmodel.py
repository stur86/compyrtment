import numpy as np
import warnings
from collections import OrderedDict, namedtuple
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from numba import jit

FitResult = namedtuple('FitResult', ['C', 'y0', 'R2', 'MRPD', 'success'])


@jit(nopython=True)
def _gillespie(tmax, y0, C, i1, i2, i3, i4, sN, tblock=100):
    nC = len(C)
    indsC = range(nC)

    traj = np.zeros((sN, tblock, len(y0)+1))

    for i in range(sN):
        y = y0.copy()
        yext = np.concatenate((y, np.ones((1,))))
        t_s = 0
        t_i = 0
        traj[i, 0, 1:] = y
        while t_s < tmax:
            w = C*yext[i1]*yext[i2]
            Wtot = np.sum(w)
            if Wtot == 0:
                # Then it's finished
                break
            dt = -np.log(np.random.random())/Wtot
            t_s += dt
            t_i += 1
            C_i = np.where(np.random.random() < np.cumsum(w)/Wtot)[0][0]
            yext[i3[C_i]] -= 1.
            yext[i4[C_i]] += 1.
            yext[-1] = 1.

            if t_i >= traj.shape[1]:
                traj = np.concatenate((traj,
                                       np.zeros((sN, tblock, len(y0)+1))),
                                      axis=1)

            traj[i, t_i, 0] = t_s
            traj[i, t_i, 1:] = yext[:-1]

    return traj


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

        if name in self._couplings:
            raise ValueError('Coupling {0} already exists '
                             'for model'.format(name))

        self._couplings[name] = (descr, C)

        self._cdata['C'] = np.concatenate([self._cdata['C'], [C]])
        self._cdata['i'] = np.concatenate(
            [self._cdata['i'], [[i1, i2, i3, i4]]], axis=0)

    def edit_coupling_rate(self, name, C):
        """Change the coupling rate for an existing coupling

        Change the coupling rate for an existing coupling

        Arguments:
            name {str} -- Name of the coupling
            C {number} -- New value
        """

        names = list(self._couplings.keys())
        try:
            i = names.index(name)
        except ValueError:
            raise ValueError('No coupling with name {0} exists'.format(name))

        descr, _ = self._couplings[name]
        self._couplings[name] = (descr, C)
        self._cdata['C'][i] = C

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

    def d2y_dtdy0(self, y, dydy0):
        """Time derivative of the model's gradient w.r.t initial conditions

        Time derivative of the gradient of the model's state with respect
        to its initial conditions.

        Arguments:
            y {np.ndarray} -- State vector
            dydy0 {np.ndarray} -- Gradient of y w.r.t. initial conditions
        """

        C = self._cdata['C']
        nC = len(C)
        i1, i2, i3, i4 = self._cdata['i'].T

        yext = np.concatenate([y, [1]])
        dydy0ext = dydy0.reshape(-1, self._N)
        if dydy0ext.shape[0] != self._N:
            raise ValueError('Invalid size gradient passed to d2y_dtdy0')
        dydy0ext = np.concatenate(
            [dydy0ext, np.zeros((self._N, 1))], axis=1)
        d2ydtdy0 = dydy0ext*0.

        iy0 = np.repeat(np.arange(self._N), nC)
        i1t = np.tile(i1, self._N)
        i2t = np.tile(i2, self._N)
        i3t = np.tile(i3, self._N)
        i4t = np.tile(i4, self._N)

        dcy = np.tile(C, self._N)*(dydy0ext[iy0, i1t]*yext[i2t] +
                                   dydy0ext[iy0, i2t]*yext[i1t])

        np.add.at(d2ydtdy0, (iy0, i3t), -dcy)
        np.add.at(d2ydtdy0, (iy0, i4t),  dcy)

        d2ydtdy0 = d2ydtdy0[:, :-1].reshape(-1)

        return d2ydtdy0

    def integrate(self, t, y0, use_gradient=False, events=None, max_step=np.inf):
        """Integrate the ODEs of the model

        Carry out an integration of the system of ODEs representing
        the model using the scipy.integrate.solve_ivp method.

        Arguments:
            t {np.ndarray} -- Array of times for the integration
            y0 {np.ndarray} -- Starting state

        Keyword Arguments:
            use_gradient {bool} -- If True, also compute the
            gradient of the curves w.r.t parameters and initial conditions
            (default: {False})
            events {[fun(t, y)]} -- List of functions of t, y (where y is the
            array of states, or states + gradients if use_gradient is True)
            at which special events happen and are recorded, or can terminate
            integration. See scipy's documentation for solve_ivp for further
            details (default: {None})

        Returns:
            OrderedDict -- Dictionary of the computed trajectories, with the
            following keys:
                - y: trajectory of the main populations
                - dy/d([coupling name]): derivatives wrt coupling constants
                - dy/d([state name]0): derivatives wrt starting conditions
                - t: times of integration. Usually the entire model.t, but can
                be less in presence of terminating events.
                - t_events: times of events.

            The derivatives are present only if use_gradient = True. The event
            times are present only if events is not None.
        """

        if not use_gradient:
            def ode(t, y):
                return self.dy_dt(y)

            sol = solve_ivp(ode, [t[0], t[-1]], y0, t_eval=t,
                            events=events, max_step=max_step)
            traj = sol.y.T

            ans = OrderedDict({'y': traj})

        else:
            N2 = self._N**2

            def ode(t, y):
                dydt = y*0.
                dydt[:self._N] = self.dy_dt(y[:self._N])
                dydt[self._N:-N2] = self.d2y_dtdC(y[:self._N],
                                                  y[self._N:-N2])
                dydt[-N2:] = self.d2y_dtdy0(y[:self._N], y[-N2:])

                return dydt

            nC = len(self._couplings)
            y0 = np.concatenate([y0, np.zeros(nC*self._N),
                                 np.eye(self._N).reshape((-1,))])

            sol = solve_ivp(ode, [t[0], t[-1]], y0, t_eval=t,
                            events=events, max_step=max_step)
            traj = sol.y.T

            ans = OrderedDict({'y': traj[:, :self._N]})

            for i, name in enumerate(self._couplings.keys()):
                ans['dy/d({0})'.format(name)] = traj[:,
                                                     (i+1)*self._N:(i+2) *
                                                     self._N]
            i0 = self._N*(nC+1)
            for i, name in enumerate(self._states):
                ans['dy/d({0}0)'.format(name)] = traj[:,
                                                      i*self._N+i0:(i+1) *
                                                      self._N+i0]

        ans['t'] = sol.t
        if events is not None:
            ans['t_events'] = sol.t_events

        return ans

    def gillespie(self, tmax, y0, samples=1000):
        """Perform a Gillespie stochastic simulation

        Simulate the system stochastically with Gillespie's algorithm.

        Arguments:
            tmax {float} -- Maximum time for the simulation
            y0 {np.ndarray} -- Starting state

        Keyword Arguments:
            samples {number} -- Number of trajectories to sample
                                (default: {1000})

        Returns:
            OrderedDict -- Dictionary of the computed trajectories, with the
            following keys:
                - t: times at which the state is computed, for each
                     trajectory
                - y: trajectories of the main populations
        """

        # Grab coupling data
        C = self._cdata['C']
        i1, i2, i3, i4 = self._cdata['i'].T

        traj = _gillespie(tmax, y0, C, i1, i2, i3, i4, samples)
        tmax_i = np.argmax(traj[:, :, 0], axis=1)
        # Fill in the rest
        for i, tmi in enumerate(tmax_i):
            traj[i, tmi:, 0] = tmax
            traj[i, tmi:, 1:] = traj[i, tmi, None, 1:]

        ans = OrderedDict({'t': traj[:, :, 0]})
        ans['y'] = traj[:, :, 1:]

        return ans

    def fit(self, data, steps=1000, constraints={}):
        """Fit the model to data

        Fit the model's parameters to a given data set, minimising the
        Root Sum Square of the error. This modifies the couplings of the model
        instance, setting them to the fitted values.

        Arguments:
            data {list} -- A list of the data points available. Can be a list
            of dictionaries or arrays. If it's dictionaries, they must all
            have a member `t' (for the time of the point) and then can have
            members with the names of various compartments of the model. If
            it's arrays, then they must have N+1 elements for N compartments;
            the first is time, and the latter are the compartments, in order.
            Missing values can be set to NaN.

        Keyword Arguments:
            steps {number} -- Number of integration steps to split the time
            interval into (default: {1000})
            constraints {dict} -- Dictionary whose keys must be valid names of
            coupling constants or X0 with X a valid name of a state, and whose
            values must be numbers.
            These constants, or the initial conditions for those states,
            will be fixed to those values and not allowed to change.

        Returns:
            FitResult - Named tuple containing the following members:
                - C:  fitted constants for each coupling
                - y0: fitted optimal starting state
                - R2: array of R squared values for goodness of fit, one
                for each curve (can be R2 > 1 due to this)
                - MRPD: Mean Relative Percent Difference, total
                - success: if True, the fitting has converged to a final value.

        Raises:
            ValueError -- Thrown if data is empty or invalid in format.
        """

        # Start by putting data in the right format
        dN = len(data)
        if dN == 0:
            raise ValueError('No data passed to fit')

        data_raw = data
        data = []

        for d in data_raw:
            if type(d) is dict:
                if 't' not in d:
                    raise ValueError('Data in dictionary format must have a '
                                     't label')
                dv = np.zeros(self._N+1)
                dv[0] = d['t']
                for i, s in enumerate(self._states):
                    dv[i+1] = d.get(s, np.nan)
            else:
                dv = np.array(d)

            dv = np.where(dv == None, np.nan, dv)
            data.append(dv)

        data = np.array(data).astype(float)
        data = data[np.argsort(data, axis=0)[:, 0]]  # Sorting by time

        # Define the time range
        t = np.linspace(data[0, 0], data[-1, 0], steps)
        t = np.concatenate([t, data[:, 0]])
        # Now sort, and identify the data points
        tsort = np.argsort(t)
        t = t[tsort]
        # Make it unique
        t, uniq_inv = np.unique(t, return_inverse=True)
        data_i = np.where(tsort - steps >= 0)[0]

        C = self._cdata['C']
        nC = len(C)

        # Get constraints mask
        mask = np.ones(nC+self._N)
        y0 = np.ones(self._N)*np.nan
        for i, name in enumerate(self._couplings.keys()):
            if name in constraints:
                self.edit_coupling_rate(name, constraints[name])
                mask[i] = 0
        for i, name in enumerate(self._states):
            if name + '0' in constraints:
                y0[i] = constraints[name + '0']
                mask[nC+i] = 0

        # Cost function

        def costf(x):
            self._cdata['C'] = x[:nC]
            y0 = x[nC:]
            traj = self.integrate(t, y0, use_gradient=True)
            traj = np.array(list(traj.values())[:-1])
            # Rebuild original array
            traj = traj[:, uniq_inv, :]

            # Gradient?
            yref = traj[:, data_i, :]
            err = (data[:, 1:]-yref[0])
            err = np.where(np.isnan(data[:, 1:]), 0, err)
            cost = np.sum(err**2)
            return cost

        # Gradient of cost function
        def costfgrad(x):
            self._cdata['C'] = x[:nC]
            y0 = x[nC:]
            traj = self.integrate(t, y0, use_gradient=True)
            traj = np.array(list(traj.values())[:-1])
            # Rebuild original array
            traj = traj[:, uniq_inv, :]

            # Gradient?
            yref = traj[:, data_i, :]
            err = (yref[0]-data[:, 1:])
            err = np.where(np.isnan(data[:, 1:]), 0, err)
            return np.sum(2*err[None]*yref[1:, :, :]*mask[:, None, None],
                          axis=(1, 2))

        x0 = data[0, 1:]
        x0 = np.where(np.isnan(x0), 0, x0)
        x0 = np.where(np.isnan(y0), x0, y0)         # Apply constraints
        x0 = np.concatenate([self._cdata['C'], x0])
        sol = minimize(costf, x0, jac=costfgrad)

        x = sol.x
        for i, name in enumerate(self._couplings.keys()):
            self.edit_coupling_rate(name, x[i])
        y0 = x[nC:]

        traj = self.integrate(t, y0)['y']
        traj = traj[uniq_inv, :]

        # R2?
        yref = traj[data_i, :]
        err = (yref-data[:, 1:])

        dataMask = 1.-np.isnan(data[:, 1:])
        dataN = np.sum(dataMask, axis=0)
        data = np.where(np.isnan(data), 0, data)
        dataAvg = np.sum(data[:, 1:], axis=0)/dataN

        SStot = np.sum((data[:, 1:]-dataAvg[None, :])**2, axis=0)
        SSreg = np.sum((yref-dataAvg[None, :])**2, axis=0)

        fitC = {c: x[i] for i, c in enumerate(self._couplings.keys())}

        avgAbs = (np.abs(yref*dataMask)+np.abs(data[:, 1:]))/2.0
        avgAbs = np.where(avgAbs > 0, avgAbs, np.inf)

        MRPD = (np.sum(np.abs(yref*dataMask-data[:, 1:])/ avgAbs)
                / np.sum(dataN))
        
        ans = FitResult(fitC, y0, SSreg/SStot, MRPD, sol.success)

        return ans

    @staticmethod
    def make_SIR(beta=0.3, gamma=0.2):
        """Make a SIR model

        Constructor for an epidemic SIR model

        Keyword Arguments:
            beta {number} -- Infection rate (default: {0.3})
            gamma {number} -- Recovery rate (default: {0.2})

        Returns:
            CModel -- A CModel object describing a SIR model
        """

        sir = CModel('SIR')
        sir.set_coupling_rate('S*I', beta, name='beta')
        sir.set_coupling_rate('I:I=>R', gamma, name='gamma')

        return sir

    @staticmethod
    def make_SIS(beta=0.3, gamma=0.2):
        """Make a SIS model

        Constructor for an epidemic SIS model

        Keyword Arguments:
            beta {number} -- Infection rate (default: {0.3})
            gamma {number} -- Recovery rate (default: {0.2})

        Returns:
            CModel -- A CModel object describing a SIS model
        """

        sir = CModel('SI')
        sir.set_coupling_rate('S*I', beta, name='beta')
        sir.set_coupling_rate('I:I=>S', gamma, name='gamma')

        return sir

    @staticmethod
    def make_LotkaVolterra(alpha=2.0/3.0, beta=4.0/3.0, gamma=1, delta=1):
        """Make a Lotka-Volterra model

        Constructor for a Lotka-Volterra model of predation

        Keyword Arguments:
            alpha {number} -- Rate of prey growth (default: {0.666})
            beta {number} -- Rate of prey killing (default: {1.333})
            gamma {number} -- Rate of predator eating (default: {1})
            delta {number} -- Rate of predator death (default: {1})

        Returns:
            CModel -- A CModel object describing a Lotka-Volterra model
        """

        lv = CModel('Pp')
        lv.set_coupling_rate('p:=>p', alpha, 'alpha')
        lv.set_coupling_rate('p*P:p=>', beta, 'beta')
        lv.set_coupling_rate('p*P:=>P', gamma, 'gamma')
        lv.set_coupling_rate('P:P=>', delta, 'delta')

        return lv
