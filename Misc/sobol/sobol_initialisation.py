# compute sobol initialisation numbers

import random
import re
from sympy import totient
# import primitive_polynomials


dimensions_per_polynomial = {n: totient(2**n-1)/n for n in range(1, 28)}
total = dimensions_per_polynomial[1]
cumulative_counts = [total]
for deg in range(2, 28):
    count = dimensions_per_polynomial[deg]
    total += count
    cumulative_counts.append(total)





def get_polynomial_degree_from_dim(dim):
    if dim == 1:
        return '01'
    elif dim == 2:
        return '02'
    elif dim <= 4:
        return '03'
    elif dim <= 6:
        return '04'
    elif dim <= 12:
        return '05'
    elif dim <= 18:
        return '06'
    elif dim <= 36:
        return '07'


random.seed(42)
bits = 32


def read_joe_kuo(path='new-joe-kuo-6_21201.txt'):
    """
    Structure of file is: d s a [m_i]
    Where d is the Sobol sequence dimension, s is the degree of the primitive polynomial, a are the encoded coefficients
    and m_i may be one or multiple initialisation integers

    The returned object is a dictionary that maps the sobol dimension to the data used to construct the direction
    numbers of that dimension.
    """
    joe_kuo_data = {}
    with open(path, 'r') as file:
        raw_header = next(file)
        # header = re.split(r' {7}', raw_header.strip())
        for counter, line in enumerate(file):
            parts = line.strip().split()
            dim, degree, coefs = map(int, parts[:3])  # these are all single ints
            init_nbrs = list(map(int, parts[3:]))  # this is a list of ints.
            joe_kuo_data[counter+2] = [dim, degree, coefs, init_nbrs]
    return joe_kuo_data


def draw_initialisation_numbers(degree=2):
    """
    the degree is the degree of the primitive polynomial modulo 2. The degree grows much slower than the dimension
    the polynomial has degree+1 bits denoting its coefficient.
    """
    init_numbers = []
    for l in range(1, degree+1):
        random_nbr = random.random()
        while int(random_nbr * 2**l) % 2 == 0:
            random_nbr = random.random()
        init_numbers.append(int(random_nbr * 2**l) * 2**(bits-l))
    return init_numbers


def fetch_primitive_polynomial(dimension=2):
    """
    return the coefficients of the primitive polynomial and its degree
    """

    pass


def build_direction_integers(init_numbers, coefs):
    """
    Recursively build the direction numbers from the initialisation numbers and coefficients of primitive polynomials
    modulo 2.
    The length of coefs is degree+1 = g_k + 1.
    Formula taken from Jaeckel's Monte Carlo Methods in finance, page 83.
    """
    # how many recursions? Need n = bits - len(init_numbers) new direction numbers
    assert 32>=len(init_numbers) > 0
    dir_nbrs = init_numbers
    g_k = len(init_numbers)
    for l in range(g_k, bits):  # l = g_k, g_k+1, ..., 31. In the book, l=g_k+1, g_k+2, ..., 32.
        v_kl = init_numbers[l-g_k] >> g_k  # init numbers are zero-indexed - in the book they are one-indexed, so this is fine.
        for j in range(1, g_k+1):  # j = 1, ..., g_k
            # coefs in the book start at index 0 - the leading coefficient is not used
            # the direction numbers in the book start at index 1 - so the first summand has index g_k+1-1
            # in my code the direction numbers index at 0. Together
            v_kl ^= int(coefs[j]) * dir_nbrs[l-j]
        dir_nbrs.append(v_kl)
    return dir_nbrs


def decode_coefs(encoded_coef, degree):
    """
    Decode the encoded coefficients of primitive polynomials mod2
    The encoded coefficient is an integer in base 10
    To decode the integer, rewrite it in base 2 (with potential 0-buffers) add a leading and trailing 1,
    split into 0's and 1's.
    """
    middle_digits = str(bin(encoded_coef)[2:])
    if middle_digits == '0':
        # special case for dimension 2: the polynomial of degree 1 has only leading and trailing coefficients,
        # no "mantissa". so the encoded '' is decoded as '11'.
        middle_digits = ''
    while len(middle_digits) < degree - 1:
        middle_digits = f'0{middle_digits}'  # 0-padding
    return f'1{middle_digits}1'


def decode_inits(encoded_inits):
    """
    Decode the initialisation integers
    The data is a whitespace separated string of integers
    The first integer is always a 1 and is the 2^31 coefficient
    The second integer is the 2^30 coef, the third is the 2^29 coef and so on
    """
    init_nbrs = []
    for i, encoded_init in enumerate(encoded_inits):
        init_nbrs.append(2**(31-i) * encoded_init)
    return init_nbrs


def get_direction_numbers(maxDim=3):
    """
    construct direction numbers in maxDim dimensions (32 per dimensions if bits==32)
    """
    direction_numbers = []
    joe_kuo_data = read_joe_kuo()
    joe_kuo_maxdim = max(joe_kuo_data.keys())
    for dim in range(1, maxDim+1):
        if dim == 1:  # first dimension is special - no recursion.
            dir_nbrs = [2**(31-i) for i in range(bits)]
            direction_numbers.append(dir_nbrs)
            continue
        elif dim <= joe_kuo_maxdim:  # return joe-kuo direction integers
            # FIXED_INITS contains fixed initialisation numbers providing good uniformity properties.
            _dim, degree, encoded_coef, encoded_inits = joe_kuo_data[dim]
            assert dim == _dim
            coefs = decode_coefs(encoded_coef, degree)
            init_numbers = decode_inits(encoded_inits)
        else:  # joe-kuo direction integers exhausted - continue with higher order polynomials and regularity-breaking
            #    initialisation
            coefs, degree = fetch_primitive_polynomial(dim)
            init_numbers = draw_initialisation_numbers(degree)
        dir_nbrs = build_direction_integers(init_numbers, coefs)
        direction_numbers.append(dir_nbrs)
    return direction_numbers


if __name__ == '__main__':
    dir_nbrs = get_direction_numbers(maxDim=21201)
    dir_nrbs_reshaped = list(map(list, zip(*dir_nbrs)))
    for ctr, bit_nbrs in enumerate(dir_nrbs_reshaped):
        count = len(bit_nbrs)
        print(ctr+1, count)
        """
        out_path = r''
        with open(rf'{out_path}\direction_nbrs_{ctr+1}.txt', 'w') as file:
            count = len(bit_nbrs)
            print(count)
            file.write(str(bit_nbrs)[1:-1])"""
