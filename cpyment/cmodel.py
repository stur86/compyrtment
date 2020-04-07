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

        self._D1 = np.zeros((self._N, self._N))
        self._D1m = self._D1.copy().astype(bool)    # Mask

        self._D2 = np.zeros((self._N, self._N, self._N))
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

    def set_coupling_rate(self, descr, C=0, add=False):
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
            add {bool} -- If True, if any coupling with the same descriptor
                          was already specified, add C to it instead of 
                          replacing it (default: {False})
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
        if s1 is None and s2 is None:
            # Constant
            if s3 is not None:
                i3 = self._states.index(s3)
                C0 = 0 if not add else self._D0[i3]
                self._D0[i3] = -C + C0
                self._D0m[i3] = (self._D0[i3] != 0)
            if s4 is not None:
                i4 = self._states.index(s4)
                C0 = 0 if not add else self._D0[i4]
                self._D0[i4] = C + C0
                self._D0m[i4] = (self._D0[i4] != 0)
        elif s2 is None:
            # Linear
            i1 = self._states.index(s1)

            if s3 is None and s4 is None:
                # Default
                s4 = s1

            if s3 is not None:
                i3 = self._states.index(s3)
                C0 = 0 if not add else self._D1[i3, i1]
                self._D1[i3, i1] = -C + C0
                self._D1m[i3, i1] = (self._D1[i3, i1] != 0)
            if s4 is not None:
                i4 = self._states.index(s4)
                C0 = 0 if not add else self._D1[i4, i1]
                self._D1[i4, i1] = C + C0
                self._D1m[i4, i1] = (self._D1[i4, i1] != 0)
        else:
            # Quadratic
            i1 = self._states.index(s1)
            i2 = self._states.index(s2)

            if s3 is None and s4 is None:
                # Default
                s3 = s1
                s4 = s2

            if s3 is not None:
                i3 = self._states.index(s3)
                C0 = 0 if not add else self._D2[i3, i1, i2]
                self._D2[i3, i1, i2] = -C + C0
                self._D2m[i3, i1, i2] = (self._D2[i3, i1, i2] != 0)
            if s4 is not None:
                i4 = self._states.index(s4)
                C0 = 0 if not add else self._D2[i4, i1, i2]
                self._D2[i4, i1, i2] = C + C0
                self._D2m[i4, i1, i2] = (self._D2[i4, i1, i2] != 0)

    
