import numpy as np
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class LinearRegressionModel(object):
    
    def __init__(self, features=None, targets=None):
        self._features = features
        self.targets = targets
        self.coeffs = None
        self.constant = None
    
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, features):
        self._features = features
        

    # Adds a column of ones to a matrix. When calculating the weights
    # or coefficients for the matrix, this ones column will result in
    # the constant coefficient for the function
    def features_with_ones(self):
        onesArr = np.array([np.ones(self.features.shape[0])])
        return np.concatenate((onesArr.T, self.features), axis = 1)
    

    # Finds a function in the form B*X + c
    def fit(self):
        # Add column of ones for finding constant
        X = self.features_with_ones()
        

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
        w = np.linalg.lstsq(X, self.targets)[0]
        

        # Since column of ones is the first column, the first 
        # coefficient from least square is our constant
        self.constant = w[0]
        

        # Transpose to get a column vector of our coefficients
        self.coeffs = np.array([w[1:]]).T
        


    # Same as fit just no column of ones meaning no constant in our function
    def coeff_fit(self):
        X = self.features
        w = np.linalg.lstsq(X, self.targets)[0]
        
        self.constant = 0
        self.coeffs = np.array([w]).T
        
    def predict(self, args):
        if (len(args[0]) < self.features.shape[1]):
            return None
        return np.dot(args, self.coeffs) + self.constant

    def plot(self):
        return
    
class PolyRegressionModel(LinearRegressionModel):
    
    def __init__(self, features=None, targets=None, order=None):
        super(PolyRegressionModel, self).__init__(targets=targets)
        self.order = order
        if (features is not None and order == None):
            self.order = features.shape[1]
        self._orig_features = features
        self._features = self.poly_matrix_transform(features)
        
    @LinearRegressionModel.features.setter
    def features(self, features):
        self._orig_features = features
        self._features = self.poly_matrix_transform(features)
        self.order = self._features.shape[1]
    
    def set_order(self, order):
        self.order = order
        self._features = self.poly_matrix_transform(self._orig_features)

    
    # Transforms a matrix of linear values into polynomial values
    
    # For example if order is 3 and 
    # each row of our matrix has two values, x1 and x2
    # after transformation our matrix now has rows with 
    # 6 values:
    # [x1, x2, x1^2, x2^2, x1^3, x2^3]
    # def poly_matrix_transform(self, matrix):
    #     poly_matrix = np.empty((matrix.shape[0], matrix.shape[1]*self.order))
    #     for i in range(0, matrix.shape[0]):
    #         row = matrix[i]
    #         poly_row = []
    #         for j in range(0, self.order):
    #             for k in range(0, matrix.shape[1]):
    #                 poly_row.append(math.pow(row[k], j + 1))
    #         # for j in range(0, matrix.shape[1]):
    #         #     poly_row.append(math.pow(row[j], j + 1))
    #         poly_matrix[i] = np.array(poly_row)
    #     return poly_matrix
    
    def poly_matrix_transform(self, matrix):
        poly_matrix = []
        for i in range(0, matrix.shape[0]):
            row = matrix[i]
            poly_row = []
            products = []
            for k in range(0, matrix.shape[1]):
                val = row[k]
                p = []
                for j in range(0, self.order - 1):
                    p.append(math.pow(k, j + 1))        
            
        for i in range(0, matrix.shape[0]):
            row = matrix[i]
            poly_row = []
            for j in range(0, self.order):
                for k in range(0, matrix.shape[1]):
                    poly_row.append(math.pow(row[k], j + 1))
            # for j in range(0, matrix.shape[1]):
            #     poly_row.append(math.pow(row[j], j + 1))
            poly_matrix[i] = np.array(poly_row)
        return poly_matrix
    
    def predict(self, args):
        # if (len(args[0]) < self.features.shape[1]):
        #     return None
        # print self.poly_matrix_transform(args)
        return np.dot(self.poly_matrix_transform(args), self.coeffs) + self.constant
    

class MultipleLinearRegression:
    
    def __init__(self, features=None, targets=None, tests=None, tests_targets=None):
        self.features = features
        self.targets = targets
        self.tests = tests
        self.tests_targets = tests_targets
        self.model = None
        self.linear_model = None
        self.poly_model = None
        
    def parse_features_targets(self, input):
        if (self.feature_length <= 0 or self.feature_length <= 0):
            return
        self.features = np.empty((self.target_length, self.feature_length))
        self.targets = np.empty(self.target_length)
        for i in range(0, self.target_length):
            line = input.readline().split()
            if (len(line) == self.feature_length + 1):
                self.features[i] = np.array(line[:-1], dtype='|S4').astype(np.float)
                self.targets[i] = float(line[-1])
            
    
    def parse_tests(self, input):
        if (self.test_length <= 0):
            return
        self.tests = np.empty((self.test_length, self.feature_length))
        for i in range(0, self.test_length):
            line = input.readline().split()
            if (len(line) == self.feature_length):
                self.tests[i] = np.array(line, dtype='|S4').astype(np.float)
                
        self.tests_targets = np.empty(self.test_length)
        for i in range(0, self.test_length):
            line = input.readline().split()
            if (len(line) == 1):
                self.tests_targets[i] = float(line[0])
        
        
        
    def parse(self, input):
        firstLine = input.readline().split()
        if (len(firstLine) == 2):
            self.feature_length = int(firstLine[0])
            self.target_length = int(firstLine[1])
            self.parse_features_targets(input)
        else:
            return
        testLine = input.readline().split()
        if (len(testLine) == 1):
            self.test_length = int(testLine[0])
            self.parse_tests(input)
    
    def LinearModel(self):
        if (self.linear_model == None):
            self.linear_model = LinearRegressionModel(self.features, self.targets)
        self.model = self.linear_model
        
    def PolyModel(self, order = None):
        if (self.poly_model == None):
            self.poly_model = PolyRegressionModel(self.features, self.targets, order)
        if (order != None):
            self.poly_model.set_order(order)
        self.model = self.poly_model
            
    def fit(self):
        if (self.model):
            self.model.fit()
    
    def coeff_fit(self):
        if (self.model):
            self.model.coeff_fit()
    
    def test(self):
        if (self.model):
            prediction = self.model.predict(self.tests)
            print "Prediction: \n", prediction
            error = np.array([self.tests_targets]).T - prediction
            print "Error: \n", error
            score = 0 
            for i in range(0, len(error)):
                score += max(1 - max(abs(error[i][0]) / prediction[i][0] - 0.1, 0), 0)
            print "Score: ", score
    
    def predict(self):
        if (self.model):
            prediction = self.model.predict(self.tests)
            return prediction
                
    
    def coeffs(self):
        if (self.model):
            return self.model.coeffs
    
    def constant(self):
        if (self.model):
            return self.model.constant
        
    def accuracy(self):
        sum = 0
        for i in range(0, self.features.shape[0]):
            feature = np.array([self.features[i]])
            sum += abs(self.targets[i] - self.model.predict(feature)[0][0]) / self.targets[i]
        return sum / self.features.shape[0]
    
    def score(self):
        score = 0
        for i in range(0, self.features.shape[0]):
            feature = np.array([self.features[i]])
            error = abs(self.targets[i] - self.model.predict(feature)[0][0]) / self.targets[i]
            score += max(1 - max(error - 0.1, 0), 0)
        return score
        
    def plot_input(self):
        figure = pyplot.figure()
        ax = Axes3D(figure)
        features_T = self.features.T
        ax.scatter(features_T[0], features_T[1], self.targets)
        pyplot.show()
        
    def surface_plot(self, range_x, range_y):
        if (len(range_x) != 2 or len(range_y) != 2):
            return
        x = np.linspace(range_x[0], range_x[1])
        y = np.linspace(range_y[0], range_y[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(X), len(Y)))
        for i in range(0, 50):
            for j in range(0, 50):
                Z[i, j] = self.model.predict(np.array([[X[i, j], Y[i, j]]]))

        figure = pyplot.figure()
        ax = figure.gca(projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                    cmap=cm.RdBu, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        figure.colorbar(surface, shrink=0.5, aspect=5)
        
        pyplot.show()
    
    def merged_plot(self, range_x, range_y):
        if (len(range_x) != 2 or len(range_y) != 2):
            return
        x = np.linspace(range_x[0], range_x[1])
        y = np.linspace(range_y[0], range_y[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(X), len(Y)))
        for i in range(0, 50):
            for j in range(0, 50):
                Z[i, j] = self.model.predict(np.array([[X[i, j], Y[i, j]]]))

        figure = pyplot.figure()
        ax = figure.gca(projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                    cmap=cm.RdBu, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        figure.colorbar(surface, shrink=0.5, aspect=5)
        
        features_T = self.features.T
        ax.scatter(features_T[0], features_T[1], self.targets)

        pyplot.show()
        
                
def output_test(str, regr):
    print str
    print "Test ouptut: \n", regr.test()
    print "\nAccuracy using inputs: ", regr.accuracy()
    print "Score using inputs: ", regr.score(), "\n"

def test_linear(regr):
    regr.LinearModel()
    regr.fit()
    output_test("Testing Linear Regression", regr)
    regr.coeff_fit()
    output_test("Testing Linear Regression With Coeffs Only", regr)
    
def test_poly(regr, order=None):
    regr.PolyModel(order)
    regr.fit()
    regr.surface_plot((0,1), (0, 1))
    str = "Testing Poly Regression"
    if (order is not None):
        str += " With Order %d" % order
    output_test(str, regr)
    regr.coeff_fit()
    regr.surface_plot((0,1), (0, 1))
    output_test(str + " With Coeffs Only", regr)
    
        
def run_tests(input):
    regr = MultipleLinearRegression()
    regr.parse(input)
    test_linear(regr)
    regr.plot_input()
    regr.surface_plot((0, 1), (0, 1))
    test_poly(regr)
    test_poly(regr, 3)
    
if __name__ == "__main__":
    run_tests(sys.stdin)
