# Minimal demo runner for Underwriting notebook cells
# Combines the simulation and logistic fit from the notebook
import sys
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    print('IMPORT_ERROR', e)
    sys.exit(2)

# 1) Simulate data
np.random.seed(0)
n = 1000
income = np.random.normal(50, 15, size=n)
urgency = np.random.poisson(0.5, size=n)
age = np.random.normal(40, 10, size=n)
X = np.vstack([income, urgency, age]).T
# generate binary default labels with a latent score
true_w = np.array([-0.05, 0.8, 0.01])  # note: urgency increases PD
logit = X.dot(true_w) - 0.2
prob = 1/(1+np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)

# 2) Fit logistic surrogate
clf = LogisticRegression(penalty='l2', solver='liblinear')
clf.fit(X, y)
w = clf.coef_[0]
b = clf.intercept_[0]
print('fitted w =', np.round(w, 3), 'b =', np.round(b,3))

# 3) Compute PD and expected loss for a new example
example = np.array([60.0, 2.0, 30.0])
S = example.dot(w) + b
PD = 1/(1+np.exp(-S))
EAD = 10000.0
LGD = 0.4
EL = PD * EAD * LGD
print(f'Score S={S:.3f}, PD={PD:.3%}, EL={EL:.2f}')

# 4) Manipulation index and mitigation: reduce urgency contribution if M > threshold
beta = np.array([0.0, 1.0, 0.0])  # only urgency is persuasion-related

def manipulation_index(feature_vec, weights, beta):
    contrib = weights * feature_vec
    m = np.sum(beta * np.maximum(0, contrib))
    return m

M = manipulation_index(example, w, beta)
M_th = 0.5
if M > M_th:
    example_mitigated = example.copy()
    example_mitigated[1] = 0.0
    S2 = example_mitigated.dot(w) + b
    PD2 = 1/(1+np.exp(-S2))
    print(f'M={M:.3f} > {M_th}, mitigated PD={PD2:.3%}')
else:
    print(f'M={M:.3f} <= {M_th}, no mitigation')
