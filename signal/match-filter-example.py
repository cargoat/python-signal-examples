#!/usr/bin/python
'''--------------------------------------------- match-filter-example.py
* ORIGIN: Arron Cargo
* AUTHOR cargo, 2022/12/30
*
* VERSION $Id$
*
* CREATED on Ubuntu 22.04 x86_64, python 3.10.12
*
* match-filter-example : Compute and plot the response of a match filter
*
* ------------------------------------------------------------------------------'''
__version__ = '$Id$'
__author__ = 'cargo'
__created_on__ = '2022/12/30'
__modified_on__ = '$Date$'

# ---------------------------------- IMPORTS --------------------------------- #
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

import AnalyticChirp as chirp

# ------------------------------------------------------------------------------
def PlotSignal (t, X, fs, fc, title) :
  '''
  Plot the instantaneous frequency and magnitude of a modulated signal
  -----------------------------------------------------------------------'''
  factor =  fs/(2*np.pi);

  magnitude = np.abs (X)

  instantaneous_frequency = factor * np.diff (np.unwrap (np.angle(X)));

  f, (ax1, ax2) = plt.subplots (2, 1, sharex=True)

  # skip the first sample because we lost it in the diff used above
  ax1.plot (t[1:], magnitude[1:])
  ax2.plot (t[1:], instantaneous_frequency + fc)

  ax1.set_title ('Magnitude')
  ax2.set_title ('Instantaneous Frequency')

  ax1.grid (True)
  ax2.grid (True)

  f.canvas.manager.set_window_title (title)

# ------------------------------------------------------------------------------
def MatchFilterExample (dt, fs, f0, fn) :
  '''
  Compute and plot the response of a match filter
  -----------------------------------------------------------------------'''
  print (dt, fs, f0, fn)

  # create a time vector
  t = np.linspace (0, dt, int(dt*fs))

  # set the carrier frequency halfway between fmin and fmax
  fmin = min(fn, f0)
  fmax = max(fn, f0)
  fc = (fmax - fmin)/2 + fmin

  # create our signal
  S = chirp.anal_chirp (t, f0, dt, fn, fc, 'linear');

  # we could just call correlate, but we want to show the steps

  # time reverse and conjugate the signal
  replica = np.conj (S[::-1])

  # window the replica to help with range lobes
  # we use a tukey to maximize bandwidth
  window = sig.windows.taylor (replica.size)
  replica *= window

  # scale the replica for unity signal gain
  replica /= np.sum (np.abs(replica))

  Sh = np.convolve (S, replica, 'same')

  PlotSignal (t, S, fs, fc, 'Signal')
  PlotSignal (t, replica, fs, fc, 'Replica')

  # note that a time shift of the peak is intentional here for easier plotting
  plt.figure ('Matched Filtered Signal')
  plt.plot (t, 20*np.log10(abs(Sh)), '.-')
  plt.grid (True)
  plt.xlabel ('Time')
  plt.ylabel ('dB')

  plt.show()

# ------------------------------------------------------------------------------
if __name__ == '__main__':

  import argparse
  import sys

  parser = argparse.ArgumentParser(
                    prog=sys.argv[0],
                    description='replica correlation example',
                    epilog='Compute and plot the response of a match filter')

  parser.add_argument ('-r', '--sample-rate-hz', type=float, default=100e3)
  parser.add_argument ('-a', '--chirp-start-hz', type=float, default=25e3)
  parser.add_argument ('-b', '--chirp-stop-hz', type=float, default=50e3)
  parser.add_argument ('-t', '--duration-sec', type=float, default=1.0)

  args = parser.parse_args()

  MatchFilterExample (args.duration_sec, args.sample_rate_hz, \
                      args.chirp_start_hz, args.chirp_stop_hz)
