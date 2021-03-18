import argparse
import numpy as np
import matplotlib.pyplot as plt

def pi_estimator(n_samples):
  est = list()
  n_inside = 0; n_all = 0
  for i in range(n_samples):
    x = np.random.uniform(-1.0, 1.0)
    y = np.random.uniform(-1.0, 1.0)
    dist = np.sqrt(np.power(x, 2) + np.power(y, 2))
    if dist <= 1.0:
      n_inside += 1
    n_all += 1
    est.append(4.0 * np.float(n_inside) / np.float(n_all))
  return np.array(est)

def pi_estimator_integral(n_samples):
  est = list()
  e = 0.0
  for i in range(n_samples):
    x = np.random.uniform(0.0, 1.0)
    y = 4.0 * np.sqrt(1.0 - np.power(x, 2))
    e += (y - e) / (np.float(i) + 1.0)
    est.append(e)
  return np.array(est)

def pi_estimator_markov_chain(n_samples, step=0.1):
  est = list()
  n_inside = 0; n_all = 0; n_accepted = 0
  x = np.random.uniform(-1.0, 1.0)
  y = np.random.uniform(-1.0, 1.0)
  for i in range(n_samples):
    dx = np.random.uniform(-step, step)
    dy = np.random.uniform(-step, step)
    xn = x + dx
    yn = y + dy
    if abs(xn) < 1.0 and abs(yn) < 1.0:
      x = xn
      y = yn
      n_accepted += 1
    dist = np.sqrt(np.power(x, 2) + np.power(y, 2))
    if dist <= 1.0:
      n_inside += 1
    n_all += 1
    est.append(4.0 * np.float(n_inside) / np.float(n_all))
  acc_ratio = np.float(n_accepted) / np.float(n_all)
  return np.array(est)

def run(n_runs, n_samples, func):
  runs = []
  for i in range(n_runs):
    e = func(n_samples)
    runs.append(e)
  return np.mean(runs, axis=0), np.std(runs, axis=0)

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='Implements pi estimators.')
parser.add_argument('--n_samples', type=int, help='number of samples',
                    required=True)
parser.add_argument('--n_runs', type=int, help='number of runs', required=True)
args = parser.parse_args()

n_runs = args.n_runs
n_samples = args.n_samples
func = pi_estimator_markov_chain

mean, std = run(n_runs, n_samples, func)

plt.fill_between(x=range(n_samples), y1=mean-std, y2=mean+std, label='mean $\pm$ std')
plt.axhline(y=np.pi, color='k', linestyle='--', label='$\pi$')
plt.ylim([np.pi-0.5, np.pi+0.5])
plt.xlim([0, 10000])
plt.xlabel('step')
plt.ylabel('estimate of $\pi$')
plt.legend()
plt.show()
