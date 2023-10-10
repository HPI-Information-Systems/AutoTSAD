from __future__ import annotations

import abc
from typing import Optional, Tuple, Type

import numpy as np
import scipy


def _all_anomalous(sequence: np.ndarray) -> np.ndarray:
    return np.ones_like(sequence, dtype=np.bool_)


class AnomalyTransform:
    """Base class for anomaly transforms.

    See Also
    --------
    holistic_tsad.anomaly.transforms.LocalPointOutlierTransform
    holistic_tsad.anomaly.transforms.CompressTransform
    holistic_tsad.anomaly.transforms.StretchTransform
    holistic_tsad.anomaly.transforms.NoiseTransform
    holistic_tsad.anomaly.transforms.SmoothingTransform
    holistic_tsad.anomaly.transforms.HMirrorTransform
    holistic_tsad.anomaly.transforms.VMirrorTransform
    holistic_tsad.anomaly.transforms.ScaleTransform
    holistic_tsad.anomaly.transforms.ReplaceWithGutenTAGPattern
    """

    def __init__(self,
                 strength: float = 0.5,
                 random_state: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None):
        assert 0 <= strength <= 1, "strength must be between 0 and 1"
        self.strength = strength
        if rng is None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = rng

    @abc.abstractmethod
    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the anomaly transform to the given subsequence.

        Parameters
        ----------
        subsequence : np.ndarray
            The original subsequence used to generate the anomaly.

        Returns
        -------
        np.ndarray
            The transformed subsequence (anomaly).
        np.ndarray
            The anomaly labels for the transformed subsequence.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        """Return the parameters of the beta distribution that is used to sample the strength parameter.

        Returns
        -------
        Tuple[float, float]
            The parameters of the beta distribution.
        """
        ...

    @staticmethod
    def get_factory(anomaly_type: str) -> Type[AnomalyTransform]:
        if anomaly_type == "outlier":
            return LocalPointOutlierTransform
        elif anomaly_type == "compress":
            return CompressTransform
        elif anomaly_type == "stretch":
            return StretchTransform
        elif anomaly_type == "noise":
            return NoiseTransform
        elif anomaly_type == "smoothing":
            return SmoothingTransform
        elif anomaly_type == "hmirror":
            return HMirrorTransform
        elif anomaly_type == "vmirror":
            return VMirrorTransform
        elif anomaly_type == "scale":
            return ScaleTransform
        elif anomaly_type == "pattern":
            return ReplaceWithGutenTAGPattern
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    @staticmethod
    def sample_strength(anomaly_type: str, rng: np.random.Generator) -> float:
        cls = AnomalyTransform.get_factory(anomaly_type)
        alpha, beta = cls.strength_beta_distribution_params()
        strength = rng.beta(alpha, beta)
        while np.abs(strength - 0.5) < 0.05:
            strength = rng.beta(alpha, beta)
        return strength


class LocalPointOutlierTransform(AnomalyTransform):
    """Local point outlier transformation.

    Takes the subsequence as the context for computing a local point outlier. The ``strength`` parameter determines the
    height of the injected point outlier. Depending on the local context, the outlier either is a local maximum or
    minimum.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import LocalPointOutlierTransform
    >>> transform = LocalPointOutlierTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([0., 3.25, 2., 3., 4., 3., 2., 1., 0., 1.])

    Parameters
    ----------
    strength : float
        Relative amplitude difference within the context window. A ``strength`` of 1 means that the point outlier has
        1.25 the height of the maximum/minimum value within the subsequence. A ``strength`` of 0 means that the point
        outlier has 0.25 times the height (it must always be visible, otherwise there would be no anomaly!).

    random_state : int, optional
        Random state for reproducibility. Either ``random_state`` or ``rng`` can be specified.

    rng : np.random.Generator, optional
        Random number generator. Either ``random_state`` or ``rng`` can be specified.

    Returns
    -------
    subsequence : np.ndarray
        Subsequence with injected point outlier.
    labels : np.ndarray
        Anomaly labels for the subsequence. Just the point outlier is labeled as an anomaly.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # determine outlier position
        margin = 0.1 * len(subsequence)
        idx = self.rng.integers(margin, len(subsequence) - margin)

        # determine if the point outlier is a local maximum or minimum
        _max = np.abs(subsequence[idx] - subsequence.max())
        _min = np.abs(subsequence[idx] - subsequence.min())
        if _max < _min:
            outlier = -_min
        else:
            outlier = _max

        # compute the height of the point outlier
        height = (self.strength + 0.25) * outlier
        # print(f"Point outlier ({'maximum' if np.sign(outlier) == 1 else 'minimum'}): position={idx}, add_height={height}")

        # inject the point outlier
        subsequence = subsequence.copy()
        subsequence[idx] += height

        # create anomaly labels
        labels = np.zeros(len(subsequence), dtype=np.bool_)
        labels[idx] = True

        return subsequence, labels

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 2, 2  # close to normal distribution


class CompressTransform(AnomalyTransform):
    """Compress transformation.

    Compresses the subsequence by a factor. The ``strength`` parameter determines the compression factor. A
    ``strength`` of 1 means that the subsequence is compressed by a factor of 4 (0.25 times the original length) and
    a ``strength`` of 0 means that the subsequence is compressed by a factor of 1.33 (0.75 times the original length).

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import CompressTransform
    >>> transform = CompressTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([0., 2.5, 3., 0.5, 1.])

    Parameters
    ----------
    strength : float
        Relative compression factor. A ``strength`` of 1 means that the subsequence is compressed by a factor of 4
        (0.25 times the original length) and a ``strength`` of 0 means that the subsequence is compressed by a factor
        of 1.33 (0.75 times the original length).

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Compressed subsequence. Has a different length!
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        factor = self.factor(self.strength)
        # print(f"Compression factor={factor}")
        new_length = int(len(subsequence) * factor)

        # compress the subsequence
        subsequence = subsequence.copy()
        subsequence = np.interp(np.linspace(0, len(subsequence)-1, new_length), np.arange(len(subsequence)), subsequence)
        # subsequence = interp1d(np.arange(len(subsequence)), subsequence, kind="cubic")(np.linspace(0, len(subsequence)-1, new_length))
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1, 1  # uniform distribution

    @staticmethod
    def factor(strength: float) -> float:
        """Compute the compression factor for a given ``strength`` between 0.25 and 0.75.

        Parameters
        ----------
        strength : float
            Relative compression factor. A ``strength`` of 1 means that the subsequence is compressed by a factor of 4
            (0.25 times the original length) and a ``strength`` of 0 means that the subsequence is compressed by a
            factor of 1.33 (0.75 times the original length).

        Returns
        -------
        factor : float
            Compression factor.
        """
        return 0.25 + (1 - strength) * 0.5


class StretchTransform(AnomalyTransform):
    """Stretch transformation.

    Stretches the subsequence by a factor. The ``strength`` parameter determines the stretch factor. A ``strength`` of 1
    means that the subsequence is stretched by a factor of 2 and a ``strength`` of 0 means that the subsequence is
    stretched by a factor of 1.25.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import StretchTransform
    >>> transform = StretchTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([0.        , 0.66666667, 1.33333333, 2.        , 2.66666667,
           3.33333333, 4.        , 3.33333333, 2.66666667, 2.        ,
           1.33333333, 0.66666667, 0.        , 0.66666667, 1.        ,
           1.        ])

    Parameters
    ----------
    strength : float
        Relative stretch factor. A ``strength`` of 1 means that the subsequence is stretched by a factor of 2 and a
        ``strength`` of 0 means that the subsequence is stretched by a factor of 1.25.

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Stretched subsequence. Has a different length!
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        factor = self.factor(self.strength)
        # print(f"Stretch factor={factor}")
        new_length = int(len(subsequence) * factor)

        # stretch the subsequence
        subsequence = subsequence.copy()
        subsequence = np.interp(np.linspace(0, len(subsequence)-1, new_length), np.arange(len(subsequence)), subsequence)

        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1, 1  # uniform distribution

    @staticmethod
    def factor(strength: float) -> float:
        """Compute the stretch factor for a given ``strength`` between 1.25 and 2.

        Parameters
        ----------
        strength : float
            Relative stretch factor. A ``strength`` of 1 means that the subsequence is stretched by a factor of 2 and a
            ``strength`` of 0 means that the subsequence is stretched by a factor of 1.25.

        Returns
        -------
        factor : float
            Stretch factor.
        """
        return 1.25 + strength * 0.75


class NoiseTransform(AnomalyTransform):
    """Noise transformation.

    Adds additive noise to the subsequence. The ``strength`` parameter determines the noise level. The noise level is
    between 0.02 and 0.25 times the standard deviation of the subsequence.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import NoiseTransform
    >>> transform = NoiseTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([ 5.41299681e-02,  8.15257134e-01,  2.13331021e+00,  3.16708200e+00,
            3.65341794e+00,  2.76868072e+00,  2.02270958e+00,  9.43822639e-01,
           -2.98455906e-03,  8.48465204e-01])

    Parameters
    ----------
    strength : float
        Relative noise level. The noise level is between 0.02 and 0.25 times the standard deviation of the subsequence.

    random_state : int, optional
        Random state for reproducibility. Either ``random_state`` or ``rng`` can be specified.

    rng : np.random.Generator, optional
        Random number generator. Either ``random_state`` or ``rng`` can be specified.

    Returns
    -------
    subsequence : np.ndarray
        Subsequence with added noise.
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # determine noise level between 0.02 and 0.25 times the standard deviation of the subsequence
        noise_level = 0.02 + self.strength * 0.23
        noise_level *= np.std(subsequence)
        # print(f"Noise level={noise_level}")

        # add noise to the subsequence
        subsequence = subsequence.copy()
        subsequence += self.rng.normal(scale=noise_level, size=len(subsequence))
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 3, 1


class SmoothingTransform(AnomalyTransform):
    """Smoothing transformation.

    Smooths the subsequence by applying a moving average filter. The ``strength`` parameter determines the smoothing
    factor. The smoothing factor is between 0.05 and 0.5 times the length of the subsequence.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import SmoothingTransform
    >>> transform = SmoothingTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([0.33333333, 1.        , 2.        , 3.        , 3.33333333,
           3.        , 2.        , 1.        , 0.66666667, 0.33333333])

    Parameters
    ----------
    strength : float
        Relative smoothing factor. The smoothing factor is between 0.05 and 0.5 times the length of the subsequence.

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Smoothed subsequence. Has a different length!
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # determine smoothing factor between 0.05 and 0.5 times the length of the subsequence
        factor = 0.05 + self.strength * 0.45
        factor *= len(subsequence)
        kernel_size = int(np.round(factor))
        # print(f"Smoothing filter size={filter_size}")

        # smooth the subsequence
        subsequence = subsequence.copy()
        # gaussian filter
        kernel = scipy.stats.norm.pdf(np.linspace(-5, 5, kernel_size), loc=0, scale=1)
        kernel /= np.sum(kernel)
        subsequence = np.convolve(subsequence, kernel, mode="same")
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1, 1  # uniform distribution


class VMirrorTransform(AnomalyTransform):
    """Vertical mirror transformation.

    Mirrors the subsequence vertically - also called reverse subsequence. The ``strength`` parameter has no effect.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import VMirrorTransform
    >>> transform = VMirrorTransform(random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([1., 0., 1., 2., 3., 4., 3., 2., 1., 0.])

    Parameters
    ----------
    strength : float
        Not used.

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Reversed subsequence.
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # reverse the subsequence
        subsequence = subsequence.copy()[::-1]
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1e10, 1  # almost always 1


class HMirrorTransform(AnomalyTransform):
    """Horizontal mirror transformation.

    Mirrors the subsequence horizontally. The mirror axis is the subsequence mean value. The ``strength`` parameter
    has no effect.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import HMirrorTransform
    >>> transform = HMirrorTransform(strength=0.5, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)

    Parameters
    ----------
    strength : float
        Not used.

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Subsequence with mirrored values.
    labels : np.ndarray
        All values are anomalous.
    """

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # mirror the subsequence
        subsequence = subsequence.copy()
        subsequence = 2 * np.mean(subsequence) - subsequence
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1e10, 1  # almost always 1


class ScaleTransform(AnomalyTransform):
    """Scale transformation.

    Scales the subsequence by a factor between 0.5 and 1.5. The ``strength`` parameter determines the scaling factor.

    Examples
    --------
    >>> import numpy as np
    >>> from autotsad.system.anomaly.transforms import ScaleTransform
    >>> transform = ScaleTransform(strength=0.4, random_state=42)
    >>> subsequence = np.array([0., 1., 2., 3., 4., 3., 2., 1., 0., 1.])
    >>> transform(subsequence)
    array([0. , 0.9, 1.8, 2.7, 3.6, 2.7, 1.8, 0.9, 0. , 0.9])

    Parameters
    ----------
    strength : float
        Relative scaling factor. The scaling factor is between 0.5 and 1.5.

    random_state : int, optional
        Not used.

    rng : np.random.Generator, optional
        Not used.

    Returns
    -------
    subsequence : np.ndarray
        Scaled subsequence.
    labels : np.ndarray
        All values are anomalous.
    """
    def __init__(self, strength: float = 0, random_state: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        assert 0 <= strength <= 1, "strength must be between 0 and 1"
        assert strength != 0.5, "strength must not be 0.5 for the ScaleTransform"
        super().__init__(strength, random_state, rng)

    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # determine scaling factor between 0.25 and 2.25
        factor = 0.25 + self.strength * 2.0
        # print(f"Scaling factor={factor}")

        # scale the subsequence
        subsequence = subsequence.copy()
        subsequence *= factor
        return subsequence, _all_anomalous(subsequence)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return .5, .5  # arcsine distribution


class ReplaceWithGutenTAGPattern(AnomalyTransform):
    """Replace subsequence with GutenTAG pattern.

    Replaces the subsequence with a GutenTAG pattern (base oscillation) of the same length. The ``strength`` parameter
    determines the base oscillation type. The following ten base oscillations are available:

    - [0.0, 0.1[ : polynomial
    - [0.1, 0.2[ : sine
    - [0.2, 0.3[ : cosine
    - [0.3, 0.4[ : sawtooth
    - [0.4, 0.5[ : square
    - [0.5, 0.6[ : dirichlet
    - [0.6, 0.7[ : mls
    - [0.7, 0.8[ : ecg
    - [0.8, 0.9[ : cylinder_bell_funnel
    - [0.9, 1.0] : random_walk

    Parameters
    ----------
    strength : float
        Determines the pattern type. Must be between 0 and 1.

    random_state : int, optional
        Random state for reproducibility. Only used for base oscillations with random components. Either
        ``random_state`` or ``rng`` can be specified.

    rng : np.random.Generator, optional
        Random number generator. Either ``random_state`` or ``rng`` can be specified.

    Returns
    -------
    subsequence : np.ndarray
        Values of the base oscillation with the same length as the input subsequence.
    labels : np.ndarray
        All values are anomalous.
    """
    def __call__(self, subsequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import gutenTAG.api.bo as gt

        n = len(subsequence)
        amplitude = max((np.max(subsequence) - np.min(subsequence))/2, 1e-6)
        ecg_method = ["simple", "ecgsyn"]

        # determine base oscillation type and generate pattern
        if self.strength < 0.1:
            pattern = gt.polynomial(n, polynomial=[0, -1/n, 1/n**2])
        elif self.strength < 0.2:
            pattern = gt.sine(n, frequency=self.rng.uniform(1., 5.), amplitude=amplitude)
        elif self.strength < 0.3:
            pattern = gt.cosine(n, frequency=self.rng.uniform(1., 5.), amplitude=amplitude)
        elif self.strength < 0.4:
            pattern = gt.sawtooth(n, frequency=self.rng.uniform(1., 5.), amplitude=amplitude, width=self.rng.uniform(0.1, 0.9))
        elif self.strength < 0.5:
            pattern = gt.square(n, frequency=self.rng.uniform(1., 5.), amplitude=amplitude, duty=self.rng.uniform(0.1, 0.9))
        elif self.strength < 0.6:
            pattern = gt.dirichlet(n, amplitude=amplitude)
        elif self.strength < 0.7:
            pattern = gt.mls(self.rng, n, amplitude=amplitude)
        elif self.strength < 0.8:
            amplitude = amplitude if ecg_method == "simple" else amplitude/4
            frequency = self.rng.uniform(1., 5.)
            method = self.rng.choice(ecg_method)
            length = max(100, n)
            pattern = gt.ecg(self.rng, length, amplitude=amplitude, frequency=frequency, ecg_sim_method=method)[:n]
        elif self.strength < 0.9:
            pattern = gt.cylinder_bell_funnel(self.rng, n)
        else:
            # always use a smoothing window of length 3 (minimum)
            pattern = gt.random_walk(self.rng, n, amplitude=amplitude, smoothing=3/n)

        return pattern, _all_anomalous(pattern)

    @staticmethod
    def strength_beta_distribution_params() -> Tuple[float, float]:
        return 1, 1  # uniform distribution
