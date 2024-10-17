import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

class BealeAlgorithm:
    """
    Implementation of Beale's variable selection algorithm.

    This class performs variable selection using Beale's algorithm, which is based on
    correlation analysis and stepwise selection.

    Attributes:
        X (np.ndarray): Standardized input features.
        y (np.ndarray): Standardized target variable.
        n_vars (int): Number of input variables.
        a (np.ndarray): Correlation matrix of X and y.
        uncond_thresholds (np.ndarray): Unconditional thresholds for each variable.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the BealeAlgorithm instance.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target variable.
        """
        self.X = StandardScaler().fit_transform(X)
        self.y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        self.n_vars = X.shape[1]
        self.a = self._compute_correlation_matrix()
        self.uncond_thresholds = self._compute_unconditional_thresholds()

    def _compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute the correlation matrix of X and y.

        Returns:
            np.ndarray: Correlation matrix.
        """
        X_with_y = np.column_stack((self.X, self.y))
        return np.corrcoef(X_with_y, rowvar=False)

    def _compute_unconditional_thresholds(self) -> np.ndarray:
        """
        Compute unconditional thresholds for each variable.

        Returns:
            np.ndarray: Array of unconditional thresholds.
        """
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
        """
        Perform pivot-in operation on the correlation matrix.

        Args:
            a (np.ndarray): Input matrix.
            q (int): Pivot index.

        Returns:
            np.ndarray: Transformed matrix after pivot-in.
        """
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
        """
        Perform pivot-out operation on the correlation matrix.

        Args:
            a (np.ndarray): Input matrix.
            q (int): Pivot index.

        Returns:
            np.ndarray: Transformed matrix after pivot-out.
        """
        b = np.zeros_like(a)
        if a[q, q] != 0:
            b[q, q] = -1 / a[q, q]
        else:
            # Handle the case where a[q, q] is zero
            b[q, q] = float('inf')  # or another appropriate value
            logging.warning(f"Division by zero encountered at b[{q}, {q}]")
        
        for j in range(len(a)):
            if j != q:
                b[j, q] = b[q, j] = -a[j, q] * b[q, q]
        for j in range(len(a)):
            for k in range(len(a)):
                if j != q and k != q:
                    b[j, k] = a[j, k] - (a[j, q] * b[q, k])
        return b

    def _select_variable(self, mlevel: np.ndarray) -> int:
        """
        Select the next variable based on the current model level.

        Args:
            mlevel (np.ndarray): Current model level.

        Returns:
            int: Index of the selected variable.
        """
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
        """
        Calculate conditional thresholds for variables.

        Args:
            mlevel (np.ndarray): Current model level for each variable.

        Returns:
            np.ndarray: Array of conditional thresholds.
        """
        epsilon = 1e-12  # Small value to avoid division by zero
        a_copy = self.a.copy()  # Create a copy of the correlation matrix
        pivoted = np.zeros(self.n_vars)  # Track which variables have been pivoted

        # Pivot in variables not in the model
        for i in range(self.n_vars):
            if mlevel[i] == -1 and a_copy[i, i] > epsilon:
                a_copy = self._pivot_in(a_copy, i)
                pivoted[i] = 1

        cond_thresholds = np.zeros(self.n_vars)  # Initialize conditional thresholds

        # Calculate conditional thresholds for pivoted variables
        for i in range(self.n_vars):
            if pivoted[i] == 1:
                temp = self._pivot_out(a_copy, i)
                cond_thresholds[i] = temp[self.n_vars, self.n_vars]

        return cond_thresholds
    


    def _update_b(self, a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
        if a[q, q] == 0:
            logging.warning(f"Division by zero encountered at b[{q}, {q}]")
            b[q, q] = float('inf')  # or another appropriate value
        else:
            b[q, q] = -1 / a[q, q]

        for j in range(self.n_vars + 1):
            if j != q:
                b[j, q] = b[q, j] = -a[j, q] * b[q, q]
                for k in range(self.n_vars + 1):
                    if k != q:
                        b[j, k] = a[j, k] - (a[j, q] * b[q, k])
        return b

    def _calculate_increase(self, i: int) -> float:
        if self.a[self.n_vars, self.n_vars] == 0:
            logging.warning(f"Division by zero encountered in increase calculation for i={i}")
            return float('inf')  # or another appropriate value
        return np.abs((self.a[self.n_vars, i]**2) / self.a[self.n_vars, self.n_vars])

    def beale(self, N: int) -> Tuple[List[int], float]:
        """
        Implements Beale's variable selection algorithm.

        Args:
            N (int): The number of variables to select.

        Returns:
            Tuple[List[int], float]: A tuple containing:
                - List[int]: Indices of selected variables.
                - float: The best residual sum of squares (RSS) achieved.

        This method performs the following steps:
        1. Initialize variables and arrays.
        2. Iteratively select variables based on their contribution to reducing RSS.
        3. Update the model level (mlevel) and perform pivot operations.
        4. Keep track of the best selection and RSS.
        5. Handle different cases based on the number of variables already selected.
        6. Adjust the model when the maximum order (maxod) changes.
        7. Perform conditional threshold calculations and updates.
        8. Continue until the stopping criteria are met.
        """

        mlevel = -np.ones(self.n_vars, dtype=int)
        noin = 0
        brssq = float('inf')
        maxod = 1
        mbest = -np.ones(self.n_vars, dtype=int)

        while True:
            selected_var = self._select_variable(mlevel)
            if selected_var != -1:
                if noin == N - 1:
                    temp = self._update_b(self.a, np.zeros_like(self.a), selected_var)
                    if temp[self.n_vars, self.n_vars] < brssq:
                        brssq = temp[self.n_vars, self.n_vars]
                        mbest = np.where(mlevel == -1, -1, mlevel)
                        mbest[selected_var] = 1
                        for i in range(self.n_vars):
                            if mlevel[i] > 0 and self.uncond_thresholds[i] > brssq:
                                mlevel[i] = 0
                elif noin < N - 1:
                    self.a = self._update_b(self.a, np.zeros_like(self.a), selected_var)
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
                        self.a = self._update_b(self.a, np.zeros_like(self.a), i)
                        noin -= 1

            min_increase = float('inf')
            min_rssq_increase_var_index = -1
            for i in range(self.n_vars):
                if mlevel[i] == maxod:
                    increase = self._calculate_increase(i)
                    if increase < min_increase:
                        min_increase = increase
                        min_rssq_increase_var_index = i

            self.a = self._update_b(self.a, np.zeros_like(self.a), min_rssq_increase_var_index)
            mlevel[min_rssq_increase_var_index] = -maxod
            noin -= 1

            cond_thresholds = self._calc_conditional_thresholds(mlevel)
            for i in range(self.n_vars):
                if mlevel[i] == -1 and cond_thresholds[i] >= brssq and noin < N - 1:
                    self.a = self._update_b(self.a, np.zeros_like(self.a), i)
                    mlevel[i] = maxod + 1
                    noin += 1

    def run(self, min_vars: int, max_vars: int) -> List[Tuple[List[int], float]]:
        """
        Runs the Beale algorithm for a range of variable counts.

        Args:
            min_vars (int): Minimum number of variables to select.
            max_vars (int): Maximum number of variables to select.

        Returns:
            List[Tuple[List[int], float]]: A list of tuples, each containing:
                - List[int]: Indices of selected variables.
                - float: The corresponding RSS for that selection.
        """
        results = []
        for N in range(min_vars, max_vars + 1):
            selected, brssq = self.beale(N)
            results.append((selected, brssq))
        return results

