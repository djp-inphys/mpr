import numpy as np
import sklearn.decomposition as sk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config_submit import config


class PCAAnalyzer:
    """
    A wrapper class for sklearn's PCA implementation with an option to disable centering.
    """
    def __init__(self, n_components='mle', center=False):
        """
        Initialize the PCAAnalyzer.
        
        Args:
        n_components (int or 'mle'): Number of components to keep. 
                                     'mle' uses automatic selection.
        center (bool): Whether to center the data before PCA.
        """
        self.n_components = n_components
        self.center = center
        self.pca = sk.PCA(n_components=n_components)
        self.mean = None

    def fit_transform(self, descs):
        """
        Fit the PCA model and transform the input data.
        
        Args:
        descs (np.array): Input data matrix.
        
        Returns:
        np.array: Transformed data matrix.
        """
        if self.center:
            return self.pca.fit_transform(descs)
        else:
            # Store the mean of the data
            self.mean = np.mean(descs, axis=0)
            
            # Center the data manually
            centered_descs = descs - self.mean
            
            # Fit PCA on centered data
            self.pca.fit(centered_descs)
            
            # Transform the original (non-centered) data
            return self.pca.transform(descs)

    def transform(self, descs):
        """
        Transform new data using the fitted PCA model.
        
        Args:
        descs (np.array): Input data matrix to transform.
        
        Returns:
        np.array: Transformed data matrix.
        """
        if self.center:
            return self.pca.transform(descs)
        else:
            # Transform the original (non-centered) data
            return self.pca.transform(descs)

class PCACovAnalyzer:
    """
    A class for performing PCA on covariance matrices.
    """
    @staticmethod
    def analyze(cv):
        """
        Perform PCA on a covariance matrix.
        
        Args:
        cv (np.array): Input covariance matrix.
        
        Returns:
        tuple: (coeff, latent) where coeff are the eigenvectors and latent are the eigenvalues.
        """
        coeff, latent, _ = np.linalg.svd(cv, full_matrices=True)
        
        totalvar = np.sum(latent)
        explained = 100 * latent / totalvar
        
        p, _ = np.shape(coeff)
        maxind = np.abs(coeff).argmax(axis=0)
        sign = np.sign(coeff[maxind, np.arange(p)])
        coeff *= sign
        
        return coeff, latent

class ScatterMatrixCalculator:
    """
    A class for calculating scatter matrices.
    """
    @staticmethod
    def mean_scatter(priors, mean_vecs):
        """
        Calculate the between-class scatter matrix.
        
        Args:
        priors (np.array): Class prior probabilities.
        mean_vecs (np.array): Mean vectors for each class.
        
        Returns:
        np.array: Between-class scatter matrix.
        """
        ndim, ncls = np.shape(mean_vecs)
        mean_vecs = np.matrix(mean_vecs.T)
        
        mn = np.mean(mean_vecs, axis=0)
        Sb = np.zeros((ndim, ndim))
        
        for cls_no in range(ncls):
            mn_vec = mean_vecs[cls_no, :].T
            diff = mn_vec - mn.T
            Sb += priors[cls_no] * (diff @ diff.T)
        
        return Sb

class KYAnalyzer:
    """
    A class for implementing the Kittler-Young (KY) feature extraction method.
    """
    def __init__(self):
        self.coeffs = None
        self.scores = None

    def fit_transform(self, data, cls):
        """
        Fit the KY model and transform the input data.
        
        Args:
        data (np.array): Input data matrix.
        cls (np.array): Class labels.
        
        Returns:
        np.array: Transformed data matrix.
        """
        num_cls = int(np.max(cls) + 1)
        no_cls = np.bincount(cls)
        priors = no_cls / np.sum(no_cls)
        
        no_dim = data.shape[1]
        Sw = np.zeros((no_dim, no_dim))
        mns = np.zeros((no_dim, num_cls))
        
        # Calculate within-class scatter matrix and class means
        for clsno in range(num_cls):
            sample = data[cls == clsno, :]
            Sw += priors[clsno] * np.cov(sample.T)
            mns[:, clsno] = np.mean(sample, axis=0)
        
        # First transformation
        U, lembda = PCACovAnalyzer.analyze(Sw)
        d = np.diag(np.sqrt(1 / lembda))
        B = U @ d
        
        # Calculate between-class scatter matrix
        Sb = ScatterMatrixCalculator.mean_scatter(priors, mns)
        
        # Second transformation
        finalTransformMatrix = B.T @ Sb @ B
        V, _ = PCACovAnalyzer.analyze(finalTransformMatrix)
        
        # Final transformation matrix
        self.coeffs = B @ V
        self.scores = data @ self.coeffs
        
        return self.scores

    def transform(self, data):
        """
        Transform new data using the fitted KY model.
        
        Args:
        data (np.array): Input data matrix to transform.
        
        Returns:
        np.array: Transformed data matrix.
        """
        if self.coeffs is None:
            raise ValueError("Fit the model first using fit_transform method.")
        return data @ self.coeffs


def load_data(file_path, delimiter=",", columns=None):
    """
    Load data from a file and return as a numpy array.
    """
    try:
        data = np.genfromtxt(file_path, skip_header=1, delimiter=delimiter)
        if columns:
            data = data[:, columns]
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def plot_3d(scores, y, title):
    """
    Create a 3D scatter plot of the first three components.
    
    Args:
    scores (np.array): Transformed data matrix.
    y (np.array): Class labels.
    title (str): Plot title.
    """
    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_zlabel('Third Component')
    ax.set_title(title)
    plt.colorbar(scatter, label='Class')
    plt.show()
    
    print("Plot displayed.")
    

def plot_3d_side_by_side(scores1, y1, title1, scores2, y2, title2):
    """
    Create two side-by-side 3D scatter plots.
    
    Args:
    scores1 (np.array): Transformed data matrix for the first plot.
    y1 (np.array): Class labels for the first plot.
    title1 (str): Title for the first plot.
    scores2 (np.array): Transformed data matrix for the second plot.
    y2 (np.array): Class labels for the second plot.
    title2 (str): Title for the second plot.
    """
    plt.close('all')
    fig = plt.figure(figsize=(18, 8))

    # First subplot (PCA)
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(scores1[:, 0], scores1[:, 1], scores1[:, 2], c=y1, cmap='viridis')
    ax1.set_xlabel('First Component')
    ax1.set_ylabel('Second Component')
    ax1.set_zlabel('Third Component')
    ax1.set_title(title1)
    plt.colorbar(scatter1, ax=ax1, label='Class')

    # Second subplot (KY)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(scores2[:, 0], scores2[:, 1], scores2[:, 2], c=y2, cmap='viridis')
    ax2.set_xlabel('First Component')
    ax2.set_ylabel('Second Component')
    ax2.set_zlabel('Third Component')
    ax2.set_title(title2)
    plt.colorbar(scatter2, ax=ax2, label='Class')

    plt.tight_layout()
    plt.show()

    print("Plots displayed side by side.")

# Example usage
def main():
    # read in CT lung cancer data
    trn_filename = config["datapath"] + "train.csv"
    trn_descs = load_data(trn_filename) 
    clsvec = trn_descs[:, 1].astype(int)
    trn_descs = trn_descs[:, 2:-1]
    

    # PCA example
    pca_analyzer = PCAAnalyzer(n_components=config["n_dim"])
    pca_scores = pca_analyzer.fit_transform(trn_descs)

    # KY example
    ky_analyzer = KYAnalyzer()
    ky_scores = ky_analyzer.fit_transform(trn_descs, clsvec)

    # Plot both PCA and KY side by side
    plot_3d_side_by_side(pca_scores, clsvec, "PCA Results", ky_scores, clsvec, "KY Results")
    print("end")

if __name__ == "__main__":
    main()
