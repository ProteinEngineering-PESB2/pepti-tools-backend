"""Transforms data module"""
from sklearn.decomposition import PCA, KernelPCA


class Transformer:
    """Transformer class"""

    def apply_pca_data(self, dataset, n_components = None):
        """Apply PCA"""
        if n_components == None:
            pca_transformer = PCA()
        else:
            pca_transformer = PCA(n_components=n_components)
        return pca_transformer.fit_transform(dataset)

    def apply_kernel_pca(self, dataset, kernel, n_components = None):
        """Apply kernel PCA"""
        if n_components == None:
            pca_transformer = KernelPCA(kernel=kernel)
        else:
            pca_transformer = KernelPCA(kernel=kernel, n_components=n_components)
        return pca_transformer.fit_transform(dataset)
