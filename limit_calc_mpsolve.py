import numpy as np
import matplotlib.pyplot as plt
import mpsolve

def plot_complex(data):
    fig, ax = plt.subplots()
    x = data.real
    y = data.imag
    ax.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    ax.set_aspect('equal')
    plt.show()

def eval_laurent(a: np.poly1d, r, s, z):
    ret = 0
    for n in range(-r, s+1):
        ret += a.coef[-(n+r+1)] * z**n
    return ret

def mps_roots(a: np.poly1d):
    ctx = mpsolve.Context()
    poly = mpsolve.MonomialPoly(ctx, len(a)-1)
    global nbr_calls
    nbr_calls += 1
    if nbr_calls % 10 == 1:
        print(nbr_calls-1, "Polynomials have been solved")
    assert a[0] != 0.0
    n = len(a) - 1
    i = 0
    while n >= 0:
        poly.set_coefficient(n, a[i].real, a[i].imag)
        n -= 1
        i += 1
    return ctx.solve(poly, algorithm=mpsolve.Algorithm.STANDARD_MPSOLVE)

def find_lim_set(a, r, s, N):
    global nbr_calls
    nbr_calls = 0
    tol = 10**(-7)
    assert len(a) == r+s+1, "inputted polynomial or wrong size, should be of length r+s+1"
    a_pol = np.poly1d(a)
    Lambdab = []
    phis = np.linspace(0, np.pi, N)
    exps = np.exp(phis * 1.0j)
    for phi in phis:
        if phi == 0:
            continue
        print("Processing phi =", phi)
        exponent = np.exp(phi * 1.0j)
        a_diff = np.zeros(r+s+1, dtype=np.complex128)
        for i in range(r+s+1):
            a_diff[i] = a[i] * np.exp(phi * (s-i) * 1.0j)

        roots = mps_roots(a - a_diff)

        for root in roots:
            if root == 0:
                continue
       
            lambdak = eval_laurent(a_pol, r, s, root)
            a_lambda = a.copy()
            a_lambda[s] -= lambdak
  
            spec_roots = mps_roots(a_lambda)
            root_abs = np.abs(spec_roots)
            root_abs.sort()
            if abs(root_abs[r-1] - root_abs[r]) < tol:
                Lambdab.append(lambdak)

    a_deriv = np.zeros(r+s+1, dtype=np.complex128)
    for n in range(-r, s+1):
        a_deriv[s-n] = a[s-n]*n

    double_roots = mps_roots(a_deriv)
    print("Processing double roots")
    for root in double_roots:
        if root == 0:
            continue
        print("root:", root)
        lambdak = eval_laurent(a_pol, r, s, root)
        a_lambda = a.copy()
        a_lambda[s] -= lambdak
        spec_roots = mps_roots(a_lambda)
        root_abs = np.abs(spec_roots)
        root_abs.sort()
        if abs(root_abs[r-1] - root_abs[r]) < tol:
            Lambdab.append(lambdak)

    return np.array(Lambdab)

def sample_run():
    N = 10000
    r = 1
    s = 3
    a = np.array([1, -3*(1+1j), 7j, 4*(1-1j), -2], dtype=np.complex128)
    Lambdab = find_lim_set(a, r, s, N)
    plot_complex(Lambdab)


if __name__ == "__main__":
    sample_run()
