import marimo

__generated_with = "0.9.32"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    from tqdm import tqdm
    import sklearn

    # Problem data.
    n = 500
    D = 1000
    np.random.seed(1)
    return D, cp, mo, n, np, sklearn, tqdm


@app.cell
def __(D, cp, n, np):
    A = np.random.randn(D,n)

    def solve_random_problem(D:int, n:int, x:cp.Variable, A, warm_start=False):
        b = cp.Parameter(D)
        
        # Construct the problem.
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),
                           [x >= 0])
        b.value = np.random.randn(D)
        prob.solve(warm_start=warm_start)
        print("Solve time:", prob.solver_stats.solve_time)
        return prob
    return A, solve_random_problem


@app.cell
def __(mo):
    mo.md(
        """
        ## Establishing a baseline
        (warmstarting with a similar problem works, as shown in the cvxpy documentation)
        https://www.cvxpy.org/tutorial/solvers/index.html?h=warm
        """
    )
    return


@app.cell
def __(A, D, cp, n, solve_random_problem):
    x = cp.Variable(n)
    solve_random_problem(D,n,x,A)
    return (x,)


@app.cell
def __(A, D, n, solve_random_problem, x):
    solve_random_problem(D,n,x,A,warm_start=True)
    return


@app.cell
def __(mo):
    mo.md("""## Running a very simple test""")
    return


@app.cell
def __(cp, np, sklearn):
    def approx_x(A, b):
        compressor = sklearn.random_projection.GaussianRandomProjection(eps=0.9)
        As = compressor.fit_transform(A)
        xs = cp.Variable(As.shape[1])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(As@xs - b)),
                           [xs >= 0])

        prob.solve(warm_start=False)
        x_big = compressor.inverse_transform(np.reshape(xs.value, (1,xs.value.shape[0])))
        return x_big
    return (approx_x,)


@app.cell
def __(D, approx_x, cp, n, np, tqdm):
    # Construct the problem.
    def test_warm():
        A = np.random.randn(D,n)
        b = np.random.randn(D)
        x_start = np.squeeze(approx_x(A, b))
        
        x = cp.Variable(n)
        x.value = x_start
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),
                           [x >= 0])
        prob.solve(warm_start=True)
        return prob

    times1 = [0 for _ in range(10)]
    for i in tqdm(range(10)):
        prob_i = test_warm()
        times1[i] = prob_i.solver_stats.solve_time
    print(f"Average solve time (warm): {np.mean(times1)}. std: {np.std(times1)}")
    return i, prob_i, test_warm, times1


@app.cell
def __(D, cp, n, np, tqdm):
    def test_cold():
        A = np.random.randn(D,n)
        b = np.random.randn(D)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),
                           [x >= 0])
        
        prob.solve(warm_start=False)
        return prob

    times2 = [0 for _ in range(10)]
    for j in tqdm(range(10)):
        prob_j = test_cold()
        times2[j] = prob_j.solver_stats.solve_time
    print(f"Average solve time (cold): {np.mean(times2)}. std: {np.std(times2)}")
    return j, prob_j, test_cold, times2


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
