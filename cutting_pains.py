from simpkleks import Simpkleks
import numpy as np

np.set_printoptions(linewidth=np.inf)


class CuttingPains():
    def __init__(self, A, b, c, eq_operator="==", verbose=True):

        self.A = A
        self.b = b
        self.c = c
        self.eq_operator = eq_operator

        self.bar_z = 0

        self.x_dim = self.c.shape[0]
        self.basis_size = self.b.shape[0]

        self.verbose = verbose
        self.max_iter = 10
        self.fp_tol = 1e-16

        self.log("INITIAL INTEGER PROBLEM")
        self.log(self)
        self.log("\n")

    def __str__(self):
        return f"A:\n{self.A}\nb:{self.b}\nc:{self.c}+{self.bar_z}\nBasis size: {self.basis_size}"

    def current_solution(self, x_bar):
        return f"x={x_bar} with objective value {self.c.T@x_bar+self.bar_z}"

    def log(self, string):
        if self.verbose:
            print(string)
        return

    def get_z(self, x_bar):
        return self.c.T@x_bar + self.bar_z

    def solve(self, initial_basis=None):
        sx = Simpkleks(self.A, self.b, self.c, verbose=True)
        sx.transform_to_eq("<=")
        self.A, self.b, self.c, self.bar_z = sx.get_model()
        self.x_dim = self.c.shape[0]

        if isinstance(initial_basis, np.ndarray) or isinstance(initial_basis, list):
            basis = initial_basis
        else:
            self.log("\nCREATING AUXILIARY PROBLEM FOR INITIAL BASIS")
            basis = np.array(sx.get_initial_basis())
            if basis.shape[0] == self.basis_size:
                self.log(
                    f"LEAVING AUXILIARY PROBLEM found feasible basis {basis}")
            else:
                # TODO give certificate
                self.log("LEAVING AUXILIARY PROBLEM")
                self.log("PROBLEM IS INFEASIBLE no feasible basis found")
                return False, False

        i = 0
        running = True
        while running:
            self.log(f"INTEGER ITERATION {i}")
            x_bar, z = sx.solve(initial_basis=basis)
            self.A, self.b, self.c, self.bar_z = sx.get_model()
            x_int = x_bar.astype(int)
            if np.all((x_bar - x_int) == 0):
                self.log(f"FOUND OPTIMAL INTEGER SOLUTION after {i} iterations")
                self.log(self.current_solution(x_bar))
                return x_bar, self.get_z(x_bar)

            nbasis = []
            for b in range(self.x_dim):
                if b not in basis:
                    nbasis.append(b)
            for ib in range(self.basis_size):
                if x_bar[ib].astype(int) - x_bar[ib] != 0:
                    LHS = np.floor(self.A[ib])
                    RHS = np.floor(self.b[ib])

                    self.b = np.append(self.b, [RHS])

                    slack_col = np.zeros((self.basis_size, 1))
                    self.A = np.hstack([self.A, slack_col])
                    slack_row = np.zeros(self.A.shape[1])
                    slack_row = np.append(LHS, [1]) 
                    self.A = np.vstack([self.A, slack_row])

                    self.c = np.append(self.c, [0])
                    # x_bar = np.append(x_bar, [1])
                    self.x_dim += 1
                    # self.basis_size += 1
                    break

            sx.set_model(self.A, self.b, self.c, self.bar_z)
            basis = np.append(basis, [nbasis[0]])
            sx.to_canonical_form(basis)
            self.A, self.b, self.c, self.bar_z = sx.get_model()
            sx.set_model(self.A, self.b, self.c, self.bar_z)

            i += 1
            if i > self.max_iter:
                self.log("TERMINATING reached max iter")
                self.log(self.current_solution(x_bar))
                return x_bar, 10
            self.log("\n")


if __name__ == "__main__":
    # OTHER PROGRAM
    # A = np.array([[1, 5, 2, 1],
    #               [-2, -9, 0, 3]])
    # b = np.array([7, -13])
    # c = np.array([1, 2, -1, 3])

    # cp = CuttingPains(A, b, c, verbose=True)
    # cp.solve()

    # INTEGER PROGRAM
    A = np.array([[1, 4],
                  [1, 1]])
    b = np.array([8, 4])
    c = np.array([2, 5])

    cp = CuttingPains(A, b, c, eq_operator="<=", verbose=True)
    cp.solve(initial_basis=None)
