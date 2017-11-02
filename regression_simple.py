import numpy as np
import math
import sys

def parse(input):
    inputArr = input.read().splitlines()
    F, N = (int(s) for s in inputArr[0].split())
    features = []
    targets = []
    for i in range(1, N + 1):
        s = inputArr[i].split()
        features.append([float(i) for i in s[0:-1]])
        targets.append([float(s[-1])])
    T = int(inputArr[N + 1].split()[0])
    t_features = [[float(i) for i in s.split()] for s in inputArr[N + 2:]]
    return np.array(features), np.array(targets), np.array(t_features)

# Adds a column of ones to a matrix. When calculating the weights
# or coefficients for the matrix, this ones column will result in
# the constant coefficient for the function
def matrix_with_ones(matrix):
    onesArr = np.array([np.ones(matrix.shape[0])])
    return np.concatenate((onesArr.T, matrix), axis=1)

def poly_transform_structure(row, order):
    base_t = []
    for i in range(0, len(row)):
        base_t.append([(i, 1)])
    poly_t = [base_t]
    
    for i in range(1, order):
        poly = []
        for j in range(0, len(base_t)):
            val = base_t[j][0]
            prev_poly = poly_t[-1]
            for k in range(0, len(prev_poly)): 
                p = prev_poly[k]
                if (p[0][0] == val[0]):
                    break
            prev_poly = prev_poly[k:]
            for p in prev_poly:
                _p = []
                if p[0][0] == val[0]:
                    _p.append((p[0][0], p[0][1] + 1))
                    _p = _p + p[1:]
                else:
                    _p = [val] + p
                poly.append(_p)
        poly_t.append(poly)
    poly_t = [item for sublist in poly_t for item in sublist]
    return poly_t

def poly_matrix_transform(matrix, order=None):
    poly_transform = poly_transform_structure(matrix[0], order)
    poly_matrix = []
    for row in matrix:
        poly_row = []
        for t in poly_transform:
            p = 1
            for _p in t:
                i = _p[0]
                ord = _p[1]
                p *= math.pow(row[i], ord)
            poly_row.append(p)
        poly_matrix.append(poly_row)
    return poly_matrix

# Finds a function in the form B*X + c
def fit(features, targets):
    # Add column of ones for finding constant]
    poly_features = np.array(poly_matrix_transform(features, 3))
    X = matrix_with_ones(poly_features)
    

    # Least square minimizes the square of the error.
    
    # The formula for the error is:
    #    error = sum(yi - xi * B)^2, i from 0 to n
    # where yi is the target for row xi. This is similar to the 
    # mean square error from neural networks.
    
    # In matrix form:
    #    error = (Y - X * B)^T (Y - X * B)
    #    (T is transpose, Matrix^T * Matrix is the Matrix^2)
    
    # To minimize we take the derivative with respect to B
    # and set to zero:
    #    X^T * Y - X^T * X * B = 0
    #    B = (X^T * X) ^ -1 * X^T * Y
    # now we have found the weights or coefficients that will
    # minimize our error
    w = np.linalg.lstsq(X, targets)[0]
    


    # Transpose to get a column vector of our coefficients
    coeffs = w[1:]
    
    # Since column of ones is the first column, the first 
    # coefficient from least square is our constant
    constant = w[0]
    
    return coeffs, constant

def predict(coeffs, constant, features):
    poly_features = np.array(poly_matrix_transform(features, 3))
    return np.dot(poly_features, coeffs) + constant
    
    

features, targets, t_features = parse(sys.stdin)
coeffs, constant = fit(features, targets)
prediction = predict(coeffs, constant, t_features)
for p in prediction:
    print p[0]

