#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:42:42 2017

@author: apm13

Principle Component Analysis playground - 
http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#drop_labels
"""
import numpy as np

np.random.seed(1) # random seed for consistency

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])

# generate 20, 3d multivariate gaussians.
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
# assert is basically "raise an exception if not true ..."
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

#
# Plot the data
#
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()

#
# Merge the data as PCA does not care about class labels.
#
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"
#
# Calc mean vector.
#
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print 'Mean Vector: \n', mean_vector, '\n' 
#
# Compute the scatter matrix - \textbf{S} = \sum_{k=1}^n (\textbf{x}_k-\textbf{m})(\textbf{x}_k-\textbf{m})^T
#  where \textbf{m} = \frac{1}{n} \sum_{k=1}^n = mean vector
#
scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
print 'Scatter Matrix: \n', scatter_matrix, '\n'

"""
# "Alternatively, instead of calculating the scatter matrix, we could also 
#calculate the covariance matrix using the in-built numpy.cov() function. 
#The equations for the covariance matrix and scatter matrix are very similar, 
#the only difference is, that we use the scaling factor 1N−11N−1 
#(here: 140−1=139140−1=139) for the covariance matrix. Thus, their eigenspaces 
#will be identical (identical eigenvectors, only the eigenvalues are scaled 
#differently by a constant factor)."
"""
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print 'Covariance Matrix: \n', cov_mat, '\n'
print(40 * '-')
"""
# Calculate eigenvectors of both covariance and scatter matrix,
# confirm that the eigenvaectors in each case are identical.
"""
# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')
"""
# Confirm that the eigenval/vector equation is satisfied Cov*eigvec = eigval * eigvec
"""
for i in range(len(eig_val_sc)):
    eigv = eig_vec_sc[:,i].reshape(1,3).T
    # side note: cool that python has an almost equal function ..
    np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
"""
# Visualise the eigenvectors.
"""
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_sc.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvectors')

plt.show()
plt.close()

"""
# Sort the eigenvectors by decreasing eigenvalues.
"""
# confirm the eigenvectors are all approximately of length one - i.e are unit vectors in the 
# feature space.
for ev in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # instead of 'assert' because of rounding errors
"""
# Then want to test the size of each eigenvectors eigenvalues and drop the smallest,
# as dropping this and moving to the smaller (2d here) subspace will lose the least amount 
# of information. "The common approach is to rank the eigenvectors from highest to 
# lowest corresponding eigenvalue and choose the top k eigenvectors."
"""
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print '\n', i[0] 

"""
# Choose the k largest eigenvectors (for this simple example we are choosing two 
# from the full 3d feature space), thus creating a new dxk (2=3, k=3 here)
# eigenvector matrix, W.
"""
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print 'Matrix W: \n', matrix_w
"""
# Can then use the new eigenvector matrix W to transform our previous values into
# the new feature subspace via y = W^T*x (x was our previous data remember).
"""
transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."

plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()
plt.close()

"""
# Now perform the PCA analysis using the in-built PCA class from matplotlib.
# https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml
# Important things to note after looking at the class attributes of the PCA class:
#   1) PCA() expects a np array as input with numrows>numcols, so we need to transpose our date.
#    
#   2) "matplotlib.mlab.PCA() keeps all dd-dimensions of the input dataset after the
#    transformation (stored in the class attribute PCA.Y), and assuming that they are already 
#    ordered (“Since the PCA analysis orders the PC axes by descending importance in terms of 
#    describing the clustering, we see that fracs is a list of monotonically decreasing values.”
#    , https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml) we just need to plot the 
#    first 2 columns if we are interested in projecting our 3-dimensional input dataset onto 
#    a 2-dimensional subspace."
# 
#   3) PCA() scales the variables to have unit variance before calculating the covariance matrix.
#      This scaling may make sense if we have different variables in different units.
"""
from matplotlib.mlab import PCA as mlabPCA

mlab_pca = mlabPCA(all_samples.T)

print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()
plt.close()

"""
Now we are going to compare the matplotlib.PCA() class with the sklearn.decomposition 
library. This lib does NOT automatically scale the variables to unit variance.


"The plot above seems to be the exact mirror image of the plot from out step by 
step approach. This is due to the fact that the signs of the eigenvectors can 
be either positive or negative, since the eigenvectors are scaled to the unit 
length 1, both we can simply multiply the transformed data by ×(−1)×(−1) to revert 
the mirror image".
"""

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)


plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()
plt.close() 

"""
The below two plots are supposed to look identical BUT they do not ... should
explore why (I have a different seed from the example online but that shouldn't 
matter).
"""
# revert to mirror image for comparison.
sklearn_transf = sklearn_transf * (-1)

# sklearn.decomposition.PCA
plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples via sklearn.decomposition.PCA')
plt.show()

# step by step PCA
plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples step by step approach')
plt.show()
