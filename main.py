import numpy as np
from numpy.linalg import norm

# from colors import bcolors
def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))  # Find diagonal coefficients
    s = np.sum(np.abs(mat), axis=1) - d  # Find row sum without diagonal
    return np.all(d > s)

# import numpy as np

def checkIfSquare(mat):
    """
    this function checks if the matrixis square.
    :param mat: matrix - type list
    :return: boolean
    """
    rows = len(mat)
    for i in mat:
        if len(i) != rows:
            return False
    return True


def isDDM(m, n):
    """
     check if given matrix is Diagonally Dominant Matrix or not.
    :param m: the matrix, type list.
    :param n: size of the matrix (nxn)
    :return: boolean
    """
    # for each row
    for i in range(0, n):

        # for each column, finding sum of each row.
        sum1 = 0
        for j in range(0, n):
            sum1 = sum1 + abs(m[i][j])

        # removing the diagonal element.
        sum1 = sum1 - abs(m[i][i])

        # checking if diagonal element is less than sum of non-diagonal element.
        if (abs(m[i][i]) < sum1):
            return False
    return True


def rowSum(row, n, x):
    """
    calculates the rows sum
    :param row: a single row from the matrix
    :param n: the row's size
    :param x: the x vector with results
    :return: the sum
    """
    sum1 = 0
    for i in range(n):
        sum1 += row[i] * x[i]
    return sum1


def checkResult(result, last_result, n, epsilon):
    """
    checking if the result is accurate enough
    :param result: the most recent result
    :param last_result: the previous result
    :param n: the size of the result vector
    :return: boolean
    """
    for i in range(n):
        if abs(result[i] - last_result[i]) > epsilon:
            return False
    return True


def Jacobi(mat, b, epsilon = 0.001):
    """
    calculating matrix to find variables vector according to yaakobi's algorithm
    :param mat: the matrix
    :param b: the result vector
    :return: the variables vector
    """

    n = len(mat)

    # check if Diagonally Dominant Matrix
    if not isDDM(mat, n):
        print("matrix is not Diagonally Dominant")


    # taking a guess: all zeros
    last_result = list()
    for i in range(n):
        last_result.append(0.0)

    result = last_result.copy()

    print("all results:\nx\t\ty\t  z")
    count = 0
    while True:
        for i in range(n):  # for each variable
            result[i] = b[i] - (rowSum(mat[i], n, last_result) - mat[i][i] * last_result[i]) # calculating sum of vars in row multiplied by resuld, but substructing the diagonal
            result[i] /= mat[i][i]


        print("i = "+str(count)+": "+str(result))
        count += 1
        if checkResult(result, last_result, n, epsilon): # end function when subtracting the result of 2 iterations is less than epsilon
            return result
        last_result = result.copy()



def gauss_seidel(A, b, X0, TOL=0.001, N=200):  #changed the TOL to be 0.001
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':
    A = np.array([[5, 1, 2],
                  [1, 6, 4],
                  [0, 3, 8]])
    b = np.array([1, 2, 3])
    print("Jacobi:\n");
    try:
        last_result = []
        last_res = Jacobi(A, b)
    except ValueError as e:
        print(str(e))
    print("\ngauss_seidel:\n");

    # A = np.array([[5, 1, 2], [1, 6, 4], [0, 3, 8]])
    # b = np.array([1, 2, 3])
    X0 = np.zeros_like(b)

    solution = gauss_seidel(A, b, X0)
    print("\nApproximate solution:", solution)

