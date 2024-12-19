import numpy as np

def gram_schmidt(vectors):
	"""Performs the Gram-Schmidt process to orthogonalize and normalize vectors."""
	orthonormal_basis = []

	for v in vectors:
	    # Step 1: Subtract projections onto all previously computed orthonormal vectors
	    for u in orthonormal_basis:
	        v -= np.dot(v, u) * u  # Subtract the projection of v onto u
	    
	    # Step 2: Normalize the resulting vector
	    norm = np.linalg.norm(v)
	    if norm > 1e-10:  # Avoid division by zero
	        v /= norm
	        orthonormal_basis.append(v)

	return np.array(orthonormal_basis)
