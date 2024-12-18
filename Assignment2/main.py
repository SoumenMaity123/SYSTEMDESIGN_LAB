import numpy as np
from Gram_Schmidt import gram_schmidt

def row_reduced_matrix(vectors):
    """Performs row reduction on the matrix and returns the row-reduced matrix."""
    A = np.array(vectors, dtype=float)
    rows, cols = A.shape
    row = 0  # Start with the first row
    
    for col in range(cols):
        if row >= rows:
            break
        
        # Find the pivot row and swap if necessary
        if A[row, col] == 0:
            for r in range(row + 1, rows):
                if A[r, col] != 0:
                    A[[row, r]] = A[[r, row]]  # Swap rows
                    break
        
        # If there is a non-zero pivot
        if A[row, col] != 0:
            # Scale the pivot row
            A[row] = A[row] / A[row, col]
            
            # Eliminate all other entries in the current column
            for r in range(rows):
                if r != row:
                    A[r] -= A[r, col] * A[row]
            row += 1
    
    return A

def print_independent_vectors(vectors, dim):

    
    # Perform row reduction
    reduced_matrix = row_reduced_matrix(vectors)
    
    # Identify non-zero rows in the row-reduced matrix
    independent_vectors = []
    for i in range(len(reduced_matrix)):
        if np.any(reduced_matrix[i] != 0):  # If the row is not zero, it's independent
            independent_vectors.append(vectors[i])
    
    return independent_vectors

dim = int(input('Enter the number of dimensions in a vector: '))
n = int(input("Enter the number of vectors: "))

vectors = []
for _ in range(n):
	print(f"Enter a vector: ")
	v = input().split() #Split the input into components

	#Convert each component to a float
	v = list(map(float, v))

	if len(v) > dim:
		v = v[:dim]
	elif len(v) < dim:
		'''Print Error'''
		print(f"Error: Vector must have exactly {dim} dimensions.")
		continue
	vectors.append(v)

# Convert list of vectors into a NumPy array
vectors = np.array(vectors, dtype=float)

# rank = np.linalg.matrix_rank(vectors)
# print(f'The rank of the matrix is: {rank}')

# if rank < dim:
# 	print(f"The number of Linearly Independent vectors must be atleast equal to the number of dimensions in the vector")
# 	exit()

# Return the list of independent vectors
independent_vectors = print_independent_vectors(vectors, dim)

#Print the linearly independent vectors
print("Linearly independent vectors:")
for v in independent_vectors:
    print(v)

orthonormal_basis = gram_schmidt(independent_vectors)
print("Orthonormal Basis")
for basis in orthonormal_basis:
	print(basis)