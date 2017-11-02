import numpy as np
import math
import sys

class Regression:
    
    def __init__(self, features, targets, ord=0, tests=None):
        self.features = features
        self.targets = targets
        self.ord = ord
        self.tests = tests
        self.num_partials = len(self.features[0])

    def matrix_with_ones(self, matrix):
        onesArr = np.array([np.ones(matrix.shape[0])])
        return np.concatenate((onesArr.T, matrix), axis=1)

    def poly_structure(self, row, order):
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

    def poly_matrix_transform(self, matrix, poly_struct):
        print poly_struct
        poly_matrix = []
        for row in matrix:
            poly_row = []
            for t in poly_struct:
                p = 1
                for _p in t:
                    i = _p[0]
                    ord = _p[1]
                    p *= math.pow(row[i], ord)
                poly_row.append(p)
            poly_matrix.append(poly_row)
        return poly_matrix
        
    def derivative_structure(self, poly_struct, coeffs, num_partials=0):
        partial_derivatives = []
        partial_coeffs = []
        for i in range(num_partials):
            struct = []
            s_coeffs = []
            for j in range(len(poly_struct)):
                p = poly_struct[j]
                coeff = coeffs[j]
                for k in range(len(p)):
                    _p = p[k]
                    if i == _p[0]:
                        ord = _p[1]
                        p[k] = (i, ord - 1)
                        coeff *= ord
                        struct.append(p)
                        s_coeffs.append(coeff)
            partial_derivatives.append(struct)
            partial_coeffs.append(s_coeffs)
        return partial_derivatives, partial_coeffs

    def differentiate(self):
        self.partial_derivatives, self.partial_coeffs = self.derivative_structure(self.poly_struct, self.coeffs, self.num_partials)
        self.derivative = []
        for deriv, coeffs in zip(self.partial_derivatives, self.partial_coeffs):
            d_f = np.array(self.poly_matrix_transform(self.features, deriv))
            self.derivative.append(np.dot(d_f, coeffs))

    # Finds a function in the form B*X + c
    def fit(self):
        # Add column of ones for finding constant]
        self.poly_struct = self.poly_structure(self.features[0], self.ord)
        self.poly_features = np.array(self.poly_matrix_transform(self.features, self.poly_struct))
        X = self.matrix_with_ones(self.poly_features)
        

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
        self.coeffs = w[1:]
        
        # Since column of ones is the first column, the first 
        # coefficient from least square is our constant
        self.constant = w[0]
        

    def predict(self, features):
        poly_features = np.array(self.poly_matrix_transform(self.poly_struct, features))
        return np.dot(poly_features, self.coeffs) + self.constant
    
    def test(self):
        if self.tests:
            self.predict(self.tests)
    
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


if __name__ == "__main__":
    features, targets, t_features = parse(sys.stdin)
    regr = Regression(features, targets, 3)
    regr.fit()
    prediction = predict(coeffs, constant, t_features)
    for p in prediction:
        print p[0]


