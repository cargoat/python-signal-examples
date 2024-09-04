#!/usr/bin/python
'''--------------------------------------------- AnalyticChirp.py

*
* AUTHOR cargo, 2022/12/31
*
* VERSION $Id$
*
* CREATED on Ubuntu 22.04 x86_64, python 3.10.12
*
* AnalyticChirp : Frequency-swept analytic cosine generator
*
* ------------------------------------------------------------------------------'''
__version__ = '$Id$'
__author__ = 'cargo'
__created_on__ = '2022/12/31'
__modified_on__ = '$Date$'

# ---------------------------------- IMPORTS --------------------------------- #
import numpy as np

# ------------------------------------------------------------------------------
def chirp_phase (t, f0, t1, f1, method='linear', vertex_zero=True):
    """
    Calculate the phase used by `chirp` to generate its output.

    See `anal_chirp` for a description of the arguments.

    """
    t = np.asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    if method in ['linear', 'lin', 'li']:
        beta = (f1 - f0) / t1
        phase = 2 * np.pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            phase = 2 * np.pi * (f0 * t + beta * t ** 3 / 3)
        else:
            phase = 2 * np.pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            phase = 2 * np.pi * f0 * t
        else:
            beta = t1 / log(f1 / f0)
            phase = 2 * np.pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            # Degenerate case: constant frequency.
            phase = 2 * np.pi * f0 * t
        else:
            # Singular point: the instantaneous frequency blows up
            # when t == sing.
            sing = -f1 * t1 / (f0 - f1)
            phase = 2 * np.pi * (-sing * f0) * log(np.abs(1 - t/sing))

    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                         " or 'hyperbolic', but a value of %r was given."
                         % method)

    return phase

# ------------------------------------------------------------------------------
def anal_chirp (t, f0, t1, f1, fc,  method='linear', vertex_zero=True):
  """Frequency-swept analytic cosine generator.

  In the following, 'Hz' should be interpreted as 'cycles per unit';
  there is no requirement here that the unit is one second.  The
  important distinction is that the units of rotation are cycles, not
  radians. Likewise, `t` could be a measurement of space instead of time.

  Parameters
  ----------
  t : array_like
      Times at which to evaluate the waveform.
  f0 : float
      Frequency (e.g. Hz) at time t=0.
  t1 : float
      Time at which `f1` is specified.
  f1 : float
      Frequency (e.g. Hz) of the waveform at time `t1`.
  fc : float
      Frequency (e.g. Hz) carrier frequency of the output chirp.
  method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
      Kind of frequency sweep.  If not given, `linear` is assumed.  See
      Notes below for more details.
  phi : float, optional
      Phase offset, in degrees. Default is 0.
  vertex_zero : bool, optional
      This parameter is only used when `method` is 'quadratic'.
      It determines whether the vertex of the parabola that is the graph
      of the frequency is at t=0 or t=t1.

  Returns
  -------
  y : ndarray
      A numpy array containing the signal evaluated at `t` with the
      requested time-varying frequency.  More precisely, the function
      returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
      (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.
  """

  # the phase at every t
  phase = chirp_phase (t, f0, t1, f1, method, vertex_zero)

  # create the carrier
  carrier = np.exp (-2.0j*np.pi*fc*t)

  # convert phase to complex cartesian and baseband
  return np.exp (1j*phase) * carrier
