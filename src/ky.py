# Libraries needed for ky (pcacov and mean_scatter)
import numpy as np
import sklearn.decomposition as sk

class PCAAnalyser:
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

class PCACovAnalyser:
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

class KYAnalyser:
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
        no_cls = np.bincount(cls.astype(int))
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
        U, lembda = PCACovAnalyser.analyze(Sw)
        d = np.diag(np.sqrt(1 / lembda))
        B = U @ d
        
        # Calculate between-class scatter matrix
        Sb = ScatterMatrixCalculator.mean_scatter(priors, mns)
        
        # Second transformation
        finalTransformMatrix = B.T @ Sb @ B
        V, _ = PCACovAnalyser.analyze(finalTransformMatrix)
        
        # Final transformation matrix
        self.coeffs = B @ V
        self.scores = data @ self.coeffs
        
        return self.coeffs, self.scores

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
