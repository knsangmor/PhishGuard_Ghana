import numpy as np
y = np.load('y_test.npy', allow_pickle=True)
p = np.load('y_prob.npy', allow_pickle=True)
ph, lg = y == 1, y == 0
print('phishing n =', ph.sum(), ' legitimate n =', lg.sum())
print('mean phishing conf =', p[ph].mean(), ' mean legit conf =', p[lg].mean())
print('phishing >0.90 =', (p[ph] > 0.90).mean())
print('legit <0.10 =', (p[lg] < 0.10).mean())