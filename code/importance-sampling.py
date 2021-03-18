import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def function(x):
	return 1.0 / (1.0 + np.exp(-x))
	
def normal_dist(mu=0, sigma=1):
	dist = stats.norm(mu, sigma)
	return dist

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='Implements pi estimators.')
parser.add_argument('--n_samples', type=int, help='number of samples',
                    required=True)
args = parser.parse_args()

n_samples = args.n_samples

mu_target = 3.0; mu_approx = 5.0
sigma_target = 1.0; sigma_approx = 1.0

px = normal_dist(mu_target, sigma_target)
qx = normal_dist(mu_approx, sigma_approx)

px_hist = np.array([np.random.normal(mu_target, sigma_target) for _ in range(n_samples)])
qx_hist = np.array([np.random.normal(mu_approx, sigma_approx) for _ in range(n_samples)])

mean_f_px = np.mean(function(px_hist))
std_f_px = np.std(function(px_hist))

mean_f_qx = np.mean(function(qx_hist))
std_f_qx = np.std(function(qx_hist))

mean_f_pxqx = np.mean(function(qx_hist)*(px.pdf(qx_hist) / qx.pdf(qx_hist)))
std_f_pxqx = np.std(function(qx_hist)*(px.pdf(qx_hist) / qx.pdf(qx_hist)))

print('f(x) from p(x): mean=', mean_f_px, 'std=', std_f_px)
print('f(x) from q(x): mean=', mean_f_qx, 'std=', std_f_qx)
print('f(x) from IS: mean=', mean_f_pxqx, 'std=', std_f_pxqx)

plt.figure()
plt.plot(np.linspace(0, 10, 100), function(np.linspace(0, 10, 100)), label='$f(x)$')
plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.legend()
plt.show()

plt.figure()
plt.hist(px_hist, bins=100, alpha=0.5, label='$p(x)$', density=True)
plt.hist(qx_hist, bins=100, alpha=0.5, label='$q(x)$', density=True)
plt.xlabel('$x$')
plt.ylabel('distribution')
plt.legend()
plt.show()

plt.figure()
plt.hist(function(px_hist), bins=100, alpha=0.5, label='$f(x)$ from $p(x)$', density=True)
plt.hist(function(qx_hist), bins=100, alpha=0.5, label='$f(x)$ from $q(x)$', density=True)
plt.xlabel('$f(x)$')
plt.ylabel('distribution')
plt.legend()
plt.show()
