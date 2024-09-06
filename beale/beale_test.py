import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class BealeAlgorithm:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = StandardScaler().fit_transform(X)
        self.y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        self.n_vars = X.shape[1]
        self.a = self._compute_correlation_matrix()
        self.uncond_thresholds = self._compute_unconditional_thresholds()

    def _compute_correlation_matrix(self) -> np.ndarray:
        X_with_y = np.column_stack((self.X, self.y))
        return np.corrcoef(X_with_y, rowvar=False)

    def _compute_unconditional_thresholds(self) -> np.ndarray:
        thresholds = np.zeros(self.n_vars)
        a_copy = self.a.copy()
        for q in range(self.n_vars):
            if a_copy[q, q] > 1e-12:
                a_copy = self._pivot_in(a_copy, q)
        for q in range(self.n_vars):
            temp = self._pivot_out(a_copy, q)
            thresholds[q] = temp[self.n_vars, self.n_vars]
        return thresholds

    @staticmethod
    def _pivot_in(a: np.ndarray, q: int) -> np.ndarray:
        b = np.zeros_like(a)
        b[q, q] = -1 / a[q, q]
        for j in range(len(a)):
            if j != q:
                b[j, q] = b[q, j] = a[j, q] * b[q, q]
        for j in range(len(a)):
            for k in range(j, len(a)):
                if j != q and k != q:
                    b[j, k] = b[k, j] = a[j, k] + (a[j, q] * b[q, k])
        return b

    @staticmethod
    def _pivot_out(a: np.ndarray, q: int) -> np.ndarray:
        b = np.zeros_like(a)
        b[q, q] = -1 / a[q, q]
        for j in range(len(a)):
            if j != q:
                b[j, q] = b[q, j] = -a[j, q] * b[q, q]
        for j in range(len(a)):
            for k in range(len(a)):
                if j != q and k != q:
                    b[j, k] = a[j, k] - (a[j, q] * b[q, k])
        return b

    def _select_variable(self, mlevel: np.ndarray) -> int:
        epsilon = 1e-12
        selected_var = -1
        most_rss_dec = 0
        a_copy = self.a.copy()
        for i in range(self.n_vars):
            if mlevel[i] == -1 and a_copy[i, i] > epsilon:
                residual_change = np.abs((a_copy[self.n_vars, i]**2) / a_copy[i, i])
                if residual_change > most_rss_dec:
                    most_rss_dec = residual_change
                    selected_var = i
        return selected_var

    def _calc_conditional_thresholds(self, mlevel: np.ndarray) -> np.ndarray:
        epsilon = 1e-12
        a_copy = self.a.copy()
        pivoted = np.zeros(self.n_vars)
        for i in range(self.n_vars):
            if mlevel[i] == -1 and a_copy[i, i] > epsilon:
                a_copy = self._pivot_in(a_copy, i)
                pivoted[i] = 1
        cond_thresholds = np.zeros(self.n_vars)
        for i in range(self.n_vars):
            if pivoted[i] == 1:
                temp = self._pivot_out(a_copy, i)
                cond_thresholds[i] = temp[self.n_vars, self.n_vars]
        return cond_thresholds

    def beale(self, N: int) -> Tuple[List[int], float]:
        mlevel = -np.ones(self.n_vars, dtype=int)
        noin = 0
        brssq = float('inf')
        maxod = 1
        mbest = -np.ones(self.n_vars, dtype=int)

        while True:
            selected_var = self._select_variable(mlevel)
            if selected_var != -1:
                if noin == N - 1:
                    temp = self._pivot_in(self.a, selected_var)
                    if temp[self.n_vars, self.n_vars] < brssq:
                        brssq = temp[self.n_vars, self.n_vars]
                        mbest = np.where(mlevel == -1, -1, mlevel)
                        mbest[selected_var] = 1
                        for i in range(self.n_vars):
                            if mlevel[i] > 0 and self.uncond_thresholds[i] > brssq:
                                mlevel[i] = 0
                elif noin < N - 1:
                    self.a = self._pivot_in(self.a, selected_var)
                    mlevel[selected_var] = maxod + 2
                    noin += 1
                    continue

            maxod = np.max(mlevel[mlevel % 2 == 1])

            if maxod < 3:
                selected = np.where(mbest >= 0)[0].tolist()
                return selected, brssq

            for i in range(self.n_vars):
                if mlevel[i] > maxod or mlevel[i] <= -maxod:
                    mlevel[i] = -1
                    if mlevel[i] > maxod:
                        self.a = self._pivot_out(self.a, i)
                        noin -= 1

            min_increase = float('inf')
            min_rssq_increase_var_index = -1
            for i in range(self.n_vars):
                if mlevel[i] == maxod:
                    increase = np.abs((self.a[self.n_vars, i]**2) / self.a[self.n_vars, self.n_vars])
                    if increase < min_increase:
                        min_increase = increase
                        min_rssq_increase_var_index = i

            self.a = self._pivot_out(self.a, min_rssq_increase_var_index)
            mlevel[min_rssq_increase_var_index] = -maxod
            noin -= 1

            cond_thresholds = self._calc_conditional_thresholds(mlevel)
            for i in range(self.n_vars):
                if mlevel[i] == -1 and cond_thresholds[i] >= brssq and noin < N - 1:
                    self.a = self._pivot_in(self.a, i)
                    mlevel[i] = maxod + 1
                    noin += 1

    def run(self, min_vars: int, max_vars: int) -> List[Tuple[List[int], float]]:
        results = []
        for N in range(min_vars, max_vars + 1):
            selected, brssq = self.beale(N)
            results.append((selected, brssq))
        return results

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    beale = BealeAlgorithm(X, y)
    results = beale.run(min_vars=2, max_vars=5)

    for N, (selected, brssq) in enumerate(results, start=2):
        print(f"Best {N}-subset: {selected}, BRSSQ: {brssq}")