import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from itertools import permutations

np.set_printoptions(linewidth=1000)


@njit
def smallest_idx_w_pos_val(array):
    # https://stackoverflow.com/a/41578614
    for idx, val in np.ndenumerate(array):
        if val > 0:
            return idx[0]
    # if nothing is found
    return -1


@njit
def smallest_idx_where_va_eq(array, v):
    for idx, val in np.ndenumerate(array):
        if val == v:
            return idx[0]
    return -1


class Simpkleks():
    def __init__(self, A, b, c, eq_operator="==", verbose=False):
        self.A = A
        self.b = b
        self.c = c
        self.eq_operator = eq_operator

        self.bar_z = 0

        self.basis_size = self.b.shape[0]
        self.x_dim = self.c.shape[0]

        self.verbose = verbose
        self.max_iter = 10
        self.fp_tol = 1e-16

        self.log("INITIAL PROBLEM")
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

    def get_model(self):
        return self.A, self.b, self.c, self.bar_z

    def set_model(self, A, b, c, bar_z):
        self.A = A
        self.b = b
        self.basis_size = self.b.shape[0]
        self.c = c
        self.bar_z = bar_z
        return

    def get_z(self, x_bar):
        return self.c.T@x_bar + self.bar_z

    def transform_to_eq(self, operator):
        if not isinstance(operator, list):
            operator = [operator for o in range(self.x_dim)]
        changed = []
        for i in range(self.basis_size):
            print(i)
            if operator[i] == "<=":
                new_col = np.zeros((self.basis_size, 1))
                new_col[i] = 1
                self.A = np.hstack([self.A, new_col])
                self.c = np.append(self.c, [0])
                changed.append("<=")
            elif operator[i] == ">=":
                new_col = np.zeros((self.basis_size, 1))
                new_col[i] = -1
                self.A = np.hstack([self.A, new_col])
                self.c = np.append(self.c, [0])
                changed.append(">=")
            elif operator[i] == "==" or operator[i] == "=":
                pass
            else:
                print("WRONG EQUALITY OPERATOR")
        if len(changed) > 0:
            print(f"Changed eq operator for {changed}")
            print(self)
        self.eq_operator = "=="
        self.x_dim += len(changed)

    def solve(self, initial_basis=None):
        print("pre", self.eq_operator)
        self.transform_to_eq(self.eq_operator)

        if isinstance(initial_basis, np.ndarray) or isinstance(initial_basis, list):
            basis = initial_basis
        else:
            self.log("\nCREATING AUXILIARY PROBLEM FOR INITIAL BASIS")
            basis = np.array(self.get_initial_basis())
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
            self.log(f"ITERATION {i}")
            self.to_canonical_form(basis)

            x_bar, terminated = self.step(basis)
            if terminated:
                # TODO maybe give certificate
                self.log(
                    f"FOUND OPTIMAL SOLUTION after {i} iterations c_n < 0")
                self.log(self.current_solution(x_bar))
                return x_bar, self.get_z(x_bar)

            new_basis, terminated = self.pick_new_basis(basis)
            if terminated:
                if new_basis == 0:
                    # TODO maybe give certificate
                    self.log(
                        f"FOUND OPTIMAL SOLUTION after {i} iterations no new basis")
                    self.log(self.current_solution(x_bar))
                    return x_bar, self.get_z(x_bar)
                elif new_basis == -1:
                    # TODO give certificate
                    self.log("TERMINATING LP is unbounded")
                else:
                    self.log("TERMINATNG unknown reasons")
                break
            basis = new_basis

            i += 1
            if i > self.max_iter:
                self.log("TERMINATING reached max iter")
                self.log(self.current_solution(x_bar))
                return x_bar, self.get_z(x_bar)
            self.log("\n")

    def find_trivial_basis(self, A):
        identity = np.eye(self.basis_size)
        for rows in permutations(range(self.x_dim), self.basis_size):
            sub_matrix = A[:, list(rows)]
            if np.array_equal(sub_matrix, identity):
                # basis = list(set(list(range(self.x_dim))).difference(rows))
                return np.array(rows)
        return None

    def get_initial_basis(self):
        try_eye = self.find_trivial_basis(self.A)

        if try_eye is not None:
            print(f"FOUND TRIVIAL BASIS {try_eye}")
            return try_eye

        # RHS non-negative
        for i in range(self.basis_size):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1
        new_c = np.zeros(self.c.shape[0]+self.basis_size)
        new_c[-self.basis_size:] = -1
        new_A = np.hstack([self.A.copy(), np.eye(self.basis_size)])
        new_b = self.b.copy()

        # aux_basis = np.arange(
        #     self.c.shape[0], self.c.shape[0]+self.basis_size, 1)
        aux_problem = Simpkleks(new_A, new_b, new_c, verbose=self.verbose)
        feasible_aux_sol, z = aux_problem.solve(
            initial_basis=np.array(range(0, self.basis_size)))
        feasible_sol = feasible_aux_sol[:-self.basis_size]
        if z > 0:
            return np.array([])
        initial_basis = np.where(feasible_sol == 0)[0]
        return initial_basis

    def to_canonical_form(self, basis):
        # P1 Ax=b -> A'x=b' with A_B=I
        A_B_inv = np.linalg.inv(self.A[:, basis])
        new_A = A_B_inv @ self.A
        new_A = np.around(new_A, decimals=7, out=None)
        new_b = A_B_inv @ self.b
        new_b = np.around(new_b, decimals=7, out=None)
        # P2 cTx -> barcTx + barz, c_B=0 and barz is constant
        A_B = self.A[:, basis]
        c_B = self.c[basis]
        y = np.linalg.inv(A_B).T @ c_B
        new_c = self.c.T-y.T@self.A
        new_c = np.around(new_c, decimals=7, out=None)
        new_bar_z = y.T @ self.b

        new_A[np.abs(new_A) <= self.fp_tol] = 0
        new_b[np.abs(new_b) <= self.fp_tol] = 0
        new_c[np.abs(new_c) <= self.fp_tol] = 0
        self.A = new_A
        self.b = new_b
        self.c = new_c
        self.bar_z += new_bar_z

        self.log("\nTO CANONICAL FORM\n" + "xxx"*10)
        self.log(f"for basis {basis}\nA_B=\n{A_B}\nA_B_inv=\n{A_B_inv}\ny={y}")
        self.log(f"NEW PROBLEM:\n{self}")

    def step(self, basis):
        nbasis = np.array(
            [x for x in range(self.c.shape[0]) if x not in basis])
        A_B = self.A[:, basis]
        c_B = self.c[basis]
        c_N = self.c[nbasis]
        # check for square and singular, i.e. basis
        assert A_B.shape[0] == A_B.shape[1]
        assert np.linalg.matrix_rank(A_B) == A_B.shape[0]
        # check for eye and c_B = 0, i.e. canonical form
        assert np.array_equal(A_B, np.eye(self.basis_size))
        assert np.sum(np.abs(c_B)) == 0

        x_bar = np.zeros(self.c.shape)
        x_bar[basis] = np.linalg.solve(A_B, self.b)

        self.log("\nSTEP\n" + "---"*10)
        self.log(f"B={basis}\nA_B=\n{A_B}\nc_B={c_B}\nx_bar={x_bar}")

        if np.max(c_N) <= 0:
            # check for optimal solution
            return x_bar, True

        if np.min(x_bar) <= 0:
            # check x_bar feasiblity
            return x_bar, False
        self.log("YOU SHOULDN'T BE HERE")

    def pick_new_basis(self, basis):
        # blands rule: always pick the smallest index
        k = smallest_idx_w_pos_val(self.c)
        # no new basis found, optimal solution found
        if k == -1:
            return 0, True

        assert k not in basis

        old_c_k = self.c[k]

        # check LP is unbounded
        if np.max(self.A[:, k]) <= 0:
            return -1, True
        x_B = (self.bar_z + self.b)/self.A[:, k]

        c_k = np.min(x_B)
        index_leaving = int(smallest_idx_where_va_eq(x_B, c_k))

        new_basis = np.copy(basis)
        new_basis[index_leaving] = k

        self.log("\nPICKING NEW BASIS\n" + "..."*10)
        self.log(f"old basis={basis}\nc={self.c}\nk={k}\nold c_k={old_c_k}")
        self.log(f"bar_z = {self.bar_z}")
        self.log(f"x_B={x_B}={self.b}/{self.A[:,k]}")
        self.log(
            f"new ck={c_k}\nindex leaving={basis[index_leaving]} index coming={k}\nnew basis={new_basis}")

        return new_basis, False


if __name__ == "__main__":
    # FEASIBLE
    # A = np.array([[1, 1, 2, 0],
    #               [0, 1, 1, 1]])
    # b = np.array([2, 5])
    # c = np.array([0, 1, 3, 0])

    # sx = Simpkleks(A, b, c, verbose=True)

    # sx.solve(initial_basis=[0, 3])

    # UNBOUNDED
    # A = np.array([[1,-2, 1, 0, 0],
    #               [0, 5,-3, 1, 0],
    #               [0, 4,-2, 0, 1]])
    # b = np.array([1,1,2])
    # c = np.array([0,-4, 3, 0, 0])

    # sx = Simpkleks(A,b,c, verbose=True)
    # sx.solve(initial_basis=[0,3,4])

    # NO INITIAL BASIS
    # A = np.array([[ 1, 5, 2, 1],
    #               [-2,-9, 0, 3]])
    # b = np.array([7, -13])
    # c = np.array([1, 2, -1, 3])

    # sx = Simpkleks(A, b, c, verbose=True)
    # sx.solve()

    # 5.1a
    # A = np.array([[ 3, 5,-6],
    #              [ 1, 3,-4],
    #              [-1, 1,-1]])
    # b = np.array([4,2,-1])
    # c = np.array([3,4,6])
    # sx = Simpkleks(A, b, c, verbose=True)
    # sx.solve()

    # 5.1a
    # A = np.array([[ 1, 0, 1,-1],
    #               [ 0, 1, 2, -1]])
    # b = np.array([1,2])
    # c = np.array([0, 0,-1, 1])
    # sx = Simpkleks(A, b, c, verbose=True)
    # sx.solve(initial_basis=[0,1])

    # 3.12
    # A = np.array([[1, -1],
    #               [1, 1]])
    # b = np.array([2, 6])
    # c = np.array([2, 1])

    # sx = Simpkleks(A, b, c, eq_operator="<=", verbose=True)

    # sx.solve(initial_basis=None)

    # Polimi 9
    # A = np.array([[2,-2,-1],
    #              [-3,3,2]])
    # b = np.array([2,3])
    # c = np.array([2, -2, 2])

    # sx = Simpkleks(A, b, c, eq_operator="<=", verbose=True)
    # sx.solve(initial_basis=None)

    # Polimi 3
    # A = np.array([[1,2,3,1],
    #              [2,1,1,2]])
    # b = np.array([3,4])
    # c = np.array([1,3,5,2])

    # sx = Simpkleks(A, b, c, eq_operator="<=", verbose=True)
    # sx.solve(initial_basis=None)

    # 3.17
    A = np.array([[1, 3, 0, 1, 1],
                 [1, 2, 0, -3, 1],
                 [-1, -4, 3, 0, 0]])
    b = np.array([2, 2, 1])
    c = np.array([-2, -3, -3, -1, 2])

    sx = Simpkleks(A, b, c, eq_operator="=", verbose=True)
    sx.solve(initial_basis=None)
