# Copyright Art Owen 2017-2021
#
# Released under BSD-3-Clause
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
# This code computes scrambled Sobol' points.
# It has been used for academic research purposes.
# It comes with no guarantees.  It might be
# much slower than other implementations.
# The primary purpose is to provide source
# for `nested uniform scrambling' (often called
# `Owen scrambling') of Sobol' points.
#
# All functionality is through a main function rsobol
#   the default is nested uniform scrambling
#   set rand = FALSE to get unrandomized Sobol'
#   set type = "mato" gets Matousek's linear scramble
#
# Basic use:
#    place files rsobol.R, fiftysobol.col, sobol_Cs.col in a directory
#    From within R in that directory run:
#      source("rsobol.R")
#      x <- rsobol(m=15,s=10,seed=1)
#   Then x will have n=2^15 scrambled Sobol' points in s=10 dimensions.
#   Repeating with the same seed gives back the same points
#   unless the underyling R pseudo-random number structure has changed.
#   There will be n rows and s columns.  For s < 50 <= 20201 use
#   x <- rsobol(fn="sobol_Cs.col",m,s,seed)
#
# Some helper functions are available with names starting with .rsobol
# That way they are available to users who want them but they don't clutter
# the namespace.  Prepending .rsobol in a few places makes some of the
# code a bit less readable as part of that tradeoff.

# This code uses generating matrices from Dirk Nuyens' magic point shop
# described in this article:
#   F.Y. Kuo & D. Nuyens. Application of quasi-Monte Carlo methods
#   to elliptic PDEs with random diffusion coefficients - a survey of
#   analysis and implementation, Foundations of Computational Mathematics,
#   16(6):1631-1696, 2016.
#
# The underlying direction numbers come from
#   Joe, Stephen, and Frances Y. Kuo.
#   "Constructing Sobol sequences with better two-dimensional projections."
#   SIAM Journal on Scientific Computing 30, no. 5 (2008): 2635-2654.
# Those papers should be cited in publications based on this code.
#
# fiftysobol.col provides generating matrices for up to 50 dimensions.
# sobol_Cs.col handles up to 21021 dimensions.
# The code reads one of them from disk at each use.
#
# The first 2^m points of rsobol(fn,m+1,s) are not the same as
# the output from rsobol(fn,m,s).
#

import numpy as np
import pandas as pd


class SobolGen:
    def __init__(self, dim, scramble=None, bits=32, seed=20251202):
        self.dim = dim
        self.scramble = scramble
        self.rand = self.scramble is not None
        self.bits = bits
        self.seed = seed

    def random_base2(self, log2_nbr_samples):
        return rsobol(fn="sobol_Cs.col", m=log2_nbr_samples, s=self.dim, rand=self.rand, type=self.scramble, M=self.bits, seed=self.seed)


def rsobol(fn="fiftysobol.col", m=10, s=5, rand=True, type="nestu", M=32, seed=20171215):
    """
    Scramble Sobol' points using either:
      - Nested uniform scramble  ("nestu")
      - Matousek linear scramble ("mato")
      - Or return plain Sobol' points when rand = False.

    # Scramble Sobol' points.  Uses nested uniform points by default.
    # fn   = file of matrix values from Dirk Nuyens' magic point shop
    # m    = log2( n ), i.e., it makes 2^n points
    # s    = dimension of points
    # rand = TRUE for randomized Sobol' (the default), FALSE for original Sobol' (use with care)
    # type = nestu (for nested uniform scramble) or mato (for Matousek's linear scramble)
    # M    = number of bits to produce; must match number of columns in fn. 32 standard
    # seed = one random seed for the whole matrix
    #
    # returns matrix in [0,1]^{n * s }of scrambled Sobol' points
    #
    """

    if m > M:
        raise RuntimeError(f"Cannot deliver 2^{m} points. Max is 2^{M}-1.")

    np.random.seed(seed)

    # Unrandomized Sobol'
    if not rand:
        return sobolpts(fn=fn, m=m, s=s, M=M)

    # bit shift scramble
    if type == "bit-shift":
        return bit_shift_main(fn=fn, m=m, s=s, M=M)

    # Matousek scramble
    if type == "mato":
        return matousek_main(fn=fn, m=m, s=s, M=M)

    # full owen scrambling (nested uniform)
    if type == "nestu":
        return owen_main(fn=fn, m=m, s=s, M=M)

    # hash-based owen scrambling
    if type == "hash-owen-bb":
        # hash-based owen using brent-burley hash
        return hash_owen_main(fn=fn, m=m, s=s, M=M, hash='bb')
    if type == "hash-owen-alt":
        # hash-based owen using psychopath v1 hash
        return hash_owen_main(fn=fn, m=m, s=s, M=M, hash='alt')
    if type == "hash-owen-final":
        # hash-based owen using psychopath final hash
        return hash_owen_main(fn=fn, m=m, s=s, M=M, hash='final')

    else:
        raise ValueError(f"Unknown scramble type: {type}. "
                         "Available options: bit-shift, nestu, mato, hash-owen-bb, hash-owen-alt, hash-owen-final")



# The next functions are helper functions for rsobol.
# Then come test functions.

def int_to_bits(x, M=32):
    # Convert an integer x into M bits
    # For large x (or small M) you get bits of x modulo 2^M
    # This does just one integer (not a vector of them).
    bits = np.zeros(M, dtype=np.uint8)
    for j in range(M):
        bits[j] = x & 1
        x >>= 1
    return bits


def sobolbits(fn="fiftysobol.col", m=8, s=20, M=32):
    # Get array of Sobol' bits - in reverse order!
    # Calls int_to_bits and sobomats
    # M must be the number of columns of data in the file named fn
    # n observations
    n = 1 << m

    # ans: shape (n, s, M)
    ans = np.zeros((n, s, M), dtype=int)

    # Sobol matrices a[j] has shape (M, M)
    a = sobomats(fn, m, s, M)

    for i in range(n):
        # bits of integer (i)
        bits_i = int_to_bits(i, M)          # 0-based, matches R: i-1 becomes i
        for j in range(s):
            # matrix multiply: M×M  times  M
            bits_j = a[j] @ bits_i
            ans[i, j, :] = bits_j % 2

    return ans


def sobomats(fn="fiftysobol.col", m=8, s=20, M=32):
    # read table (R's read.table defaults)
    # this is the direction numbers, reverse, and puts into a bit tensor
    col = pd.read_table(fn, sep=None, engine="python", header=None).values

    if s > col.shape[0]:
        raise ValueError(
            f"Not enough columns in file fn = {fn}. There should be at least s = {s}."
        )

    # output array: (s, M, M)
    a = np.zeros((s, M, M), dtype=int)

    for j in range(s):
        for k in range(M):
            a[j, :, k] = int_to_bits(col[j, k], M)
    return a


def bits_to_unif(bits):
    # Turn sequence of bits into a point in [0,1)
    # First bits are highest order
    v = 0.0
    for b in reversed(bits):
        v = (v + b) / 2.0
    return v


def bits_to_int(bitmat):
    # Convert bits b into integers, bit-reversed
    # Inverse of int_to_bits
    # This is vectorized: each row of the matrix b is a vector of bits
    #
    # bitmat: shape (n, k)
    vals = np.zeros(bitmat.shape[0], dtype=np.uint32)
    for j in range(bitmat.shape[1]):
        vals = (vals << 1) | bitmat[:, bitmat.shape[1] - 1 - j]
    return vals.astype(np.uint32)


def getpermset2(J):
    """
    Get 2^(j-1) random binary permutations for j = 1 ... J.
    Caller must set the RNG seed beforehand.

    # A nuisance is that m=0 gives J=0, requiring a list of length 0
    # that the for loop doesn't do as desired.
    # The caller will handle that corner case a different way.
    """
    ans = []

    for j in range(1, J + 1):
        length = 1 << (j - 1)  # length = 2^(j-1)
        # runif(...) > 1/2  →  Bernoulli(0.5)
        vec = (np.random.rand(length) > 0.5).astype(int)
        ans.append(vec)

    return ans


def owen_main(fn="fiftysobol.col", m=10, s=5, M=32):
    """
    m: 2**m points
    s: dimension
    M: bits, 32 default
    """
    if m == 0:
        return np.random.rand(1, s)

    n = 1 << m
    ans = np.zeros((n, s))

    # Scrambled bits via nested uniform scramble
    newbits = owen_bits(fn=fn, m=m, s=s, M=M)  # shape (n, s, M)

    # Convert bit vectors to uniforms
    for i in range(n):
        for j in range(s):
            ans[i, j] = bits_to_unif(newbits[i, j])

    return ans



def owen_bits(fn="fiftysobol.col", m=10, s=5, M=32):
    """
    m: 2**m points
    s: dimension
    M: bits, 32 default
    Nested uniform scrambling of Sobol' bits.
    # Scramble Sobol' bits; nested uniform.
    # uses sobolbits and getpermset2
    # how it works:
    # get sobol number, split into its bits
    # first bit gets a 1D permutation (0 or 1)
    # second bit gets

    """

    # 0. set seed, check at least 2 points
    if m < 1:
        raise ValueError("We need m >= 1")

    # thebits: shape (n, s, M)
    # 1. get all sobol uint32, convert to bits
    # shape: 2^m nbrs, s dims, M bits
    thebits = sobolbits(fn, m, s, M)
    newbits = thebits.copy()

    n = 1 << m  # n = 2^m
    # 2. for every dimension, scramble that dimension
    for j in range(s):
        # list of m permutation vectors
        # 2.1 get a random binary permutation each for 1, 2, ..., m bits, so for 2^0, 2^1, ..., 2^(m-1)
        theperms = getpermset2(m)

        # 3. loop over k = 1, ..., m: scramble just one bit, for all points, in a specific dimension
        for k in range(m):  # R: k = 1..m
            if k > 0:
                # bitmat: n × k slice (R: 1:(k-1))
                # 3. for dimension j, take all points, up to bit number k, e.g. the first k bits
                bitmat = thebits[:, j, :k]

                # 4. convert each row of bitmat (length k bits) to integer index
                indices = bits_to_int(bitmat)   # length n
            else:
                # R: rep(0, n)
                indices = np.zeros(n, dtype=int)

            # R indexing: theperms[[k]][1 + indices]
            # Python: theperms[k][indices]
            # because indices ∈ {0, ..., 2^(k)-1} already zero-based
            # 4. get permutation for k bits
            perm_vec = theperms[k]           # length 2^k
            # 5. do binary or with permutation. What number? the one the k bits point to.
            scrambled = (thebits[:, j, k] + perm_vec[indices]) % 2  # is this binary or?
            newbits[:, j, k] = scrambled

    # 6. If M > m, fill remaining bits with random Bernoulli(0.5)
    # why? Because we have no bits past m, each series of m bits is unique - no repetitions
    # so the distributions no longer share common bit sequences. This means it is still a nested uniform sample
    if M > m:
        rnd = (np.random.rand(n, s, M - m) > 0.5).astype(int)
        newbits[:, :, m:M] = rnd

    # in practice, I want to loop over the length of the sequence, do all dimensions and bits at once
    # there is no dependence on the dimension - can do that in inner loop
    # idea: use one draw of mrg32k3a to draw 32 bernoulli(0.5)! could be a massive speedup.
    # in principle: create a permutation tree for each dimension

    return newbits


def get_matousek2(J):
    """
    Generate Matoušek's linear scramble (base 2)
    for one Sobol dimension.

    Returns:
        M : J×J binary (0/1) matrix
        C : length-J binary (0/1) vector
    """
    # Identity matrix
    M = np.eye(J, dtype=int)

    # C = Bernoulli(0.5)
    C = (np.random.rand(J) > 0.5).astype(int)

    # Fill lower-triangular entries
    for i in range(1, J):
        for j in range(i):
            M[i, j] = int(np.random.rand() > 0.5)

    return {"M": M, "C": C}


def rsobolbits_mato(fn="fiftysobol.col", m=10, s=5, M=32):
    # workhorse function for randsobolbitsmato
    # Scramble Sobol' bits using Matousek's linear scramble
    # Uses sobolbits and get_matousek2
    if m < 1:
        raise ValueError("Need m >= 1")

    # thebits: shape (n, s, M)
    thebits = sobolbits(fn, m, s, M)
    newbits = thebits.copy()

    n = 1 << m

    for j in range(s):
        themato = get_matousek2(m)   # dict with "M" and "C"
        Mmat = themato["M"]          # shape (m, m)
        Cvec = themato["C"]          # shape (m,)

        for k in range(m):           # R loop: k = 1..m
            if k > 0:
                # bitmat: n × (k) slice (equivalent to R's [:, 1:(k-1)])
                bitmat = thebits[:, j, :k]

                # vectorized dot product with appropriate row of M
                contrib = bitmat @ Mmat[k, :k]

                newbits[:, j, k] = (thebits[:, j, k] + contrib) % 2

            # add C[k]
            newbits[:, j, k] = (newbits[:, j, k] + Cvec[k]) % 2

    # fill remaining bits with random Bernoulli(0.5)
    if M > m:
        rnd = (np.random.rand(n, s, M - m) > 0.5).astype(int)
        newbits[:, :, m:M] = rnd

    return newbits


def matousek_main(fn="fiftysobol.col",m=8,s=5,M=32):
    if m == 0:
        return np.random.rand(1, s)
    n = 1 << m
    ans = np.zeros((n, s))

    # scrambled Sobol bits (Matoušek linear scrambling)
    newbits = rsobolbits_mato(fn, m, s, M)   # shape (n, s, M)

    # convert each bit-vector into a uniform value
    for i in range(n):
        for j in range(s):
            ans[i, j] = bits_to_unif(newbits[i, j])

    return ans


def bit_shift_main(fn="fiftysobol.col",m=8,s=5,M=32):
    """
    simple bit shift scrambled sobol
    m: 2**m points
    s: dimension
    M: bits, 32 default
    """
    if m == 0:
        return np.random.rand(1, s)

    n = 1 << m
    ans = np.zeros((n, s))

    # Scrambled bits via nested uniform scramble
    newbits = bit_shift(fn=fn, m=m, s=s, M=M)  # shape (n, s, M)

    # Convert bit vectors to uniforms
    for i in range(n):
        for j in range(s):
            ans[i, j] = bits_to_unif(newbits[i, j])

    return ans


def bit_shift(fn="fiftysobol.col", m=8, s=5, M=32):
    """
    apply bit shift to sobol bits in reverse order
    """
    if m < 1:
        raise ValueError("We need m >= 1")

    # thebits: shape (n, s, M)
    # 1. get all sobol uint32, convert to bits
    # shape: 2^m nbrs, s dims, M bits
    thebits = sobolbits(fn, m, s, M)
    newbits = thebits.copy()

    # bit-shift: get one permutation per dimension, XOR it for all elements in the sequence
    perms =  (np.random.randint(0, 2, (s, M)))
    for k in range(s):
        newbits[:, k, :] = (thebits[:, k, :] + perms[None, k, :]) % 2

    return newbits


def hash_owen_main(fn="fiftysobol.col", m=8, s=5, M=32, hash='bb'):
    """
    simple bit shift scrambled sobol
    m: 2**m points
    s: dimension
    M: bits, 32 default
    """
    if m == 0:
        return np.random.rand(1, s)

    n = 1 << m
    ans = np.zeros((n, s))

    # Scrambled bits via hash-based owen scramble
    newbits = hash_owen(fn=fn, m=m, s=s, M=M, hash=hash)  # shape (n, s, M)

    # Convert bit vectors to uniforms
    for i in range(n):
        for j in range(s):
            ans[i, j] = bits_to_unif(newbits[i, j])

    return ans


def reverse32(x):
    x = ((x >> 1)  & 0x55555555) | ((x & 0x55555555) << 1)
    x = ((x >> 2)  & 0x33333333) | ((x & 0x33333333) << 2)
    x = ((x >> 4)  & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4)
    x = ((x >> 8)  & 0x00FF00FF) | ((x & 0x00FF00FF) << 8)
    x = (x >> 16) | (x << 16)
    return x & 0xFFFFFFFF


def hash_owen(fn="fiftysobol.col", m=8, s=5, M=32, hash='bb'):
    """
    apply hash-based owen scrambling to sobol bits in reverse order
    """
    if m < 1:
        raise ValueError("We need m >= 1")

    hash_func_dct = {'bb': brent_burley_hash, 'alt': psychopath_hash_original, 'final': psychopath_hash_final}
    assert hash in hash_func_dct

    # thebits: shape (n, s, M)
    # 1. get all sobol uint32, convert to bits
    # shape: 2^m nbrs, s dims, M bits
    thebits = sobolbits(fn, m, s, M)
    newbits = thebits.copy()
    n = 1 << m

    # laine-karras hash: apply hash per dimension
    perms_1 = bits_to_int(np.random.randint(0, 2, (s, M))).astype(np.uint32)
    perms_2 = bits_to_int(np.random.randint(0, 2, (s, M))).astype(np.uint32)
    for k in range(s):
        sobol_ints_rev = bits_to_int(thebits[:, k, :]).astype(np.uint32)  # these are now bit-reversed!
        # sobol_ints_rev = reverse32(sobol_ints_rev)
        sobol_ints_rev = hash_func_dct[hash](sobol_ints_rev, perms_1[k], perms_2[k])
        # output_ints = reverse32(sobol_ints_rev)
        for j in range(n):
            # TODO: speed this up
            newbits[j, k, :] = int_to_bits(sobol_ints_rev[j])
        # newbits[:, k, :] = (thebits[:, k, :] + perms[None, k, :]) % 2
    return newbits


def brent_burley_hash(sobol_ints, seed1, seed2):
    """
    laine-karras type hashing without bit reversal steps
    """
    sobol_ints += seed1
    sobol_ints ^= sobol_ints * 0x6c50b47c
    sobol_ints ^= sobol_ints * 0xb82f1e52
    sobol_ints ^= sobol_ints * 0xc7afe638
    sobol_ints ^= sobol_ints * 0x8d22f6e6
    return sobol_ints


def psychopath_hash_original(sobol_ints, seed1, seed2):
    """
    laine-karras type hashing without bit reversal steps
    """
    sobol_ints *= 0x788aeeed
    sobol_ints ^= (sobol_ints * 0x41506a02)
    sobol_ints = sobol_ints + seed1  # add seed
    sobol_ints *= (seed2 | 1)  # multiply with odd seed
    sobol_ints ^= sobol_ints * 0x7483dc64
    return sobol_ints


def psychopath_hash_final(sobol_ints, seed1, seed2):
    """
    laine-karras type hashing without bit reversal steps
    """
    sobol_ints ^= sobol_ints * 0x3d20adea
    sobol_ints += seed1
    sobol_ints *= (seed2 | 1)  # multiply with odd seed
    sobol_ints ^= sobol_ints * 0x05526c56
    sobol_ints ^= sobol_ints * 0x53a22864
    return sobol_ints


def sobolpts(fn="fiftysobol.col", m=8, s=2, M=32):
    # Plain Sobol' points
    # Caller gets them as an option to rsobol
    #
    # Warning: these points are not randomized!
    col = pd.read_csv(fn, delim_whitespace=True, header=None).values
    if s > col.shape[0]:
        raise ValueError(
            f"Not enough columns in file {fn}: need at least {s}"
        )

    # a[j, :, k] = M-bit vector encoding col[j,k]
    a = np.zeros((s, M, M), dtype=np.uint32)
    for j in range(s):
        for k in range(M):
            a[j, :, k] = int_to_bits(int(col[j, k]), M)

    n = 1 << m
    ans = np.zeros((n, s))

    for i in range(n):
        bitsi = int_to_bits(i, M)
        for j in range(s):
            # matrix-vector multiply modulo 2
            bitsj = (a[j] @ bitsi) % 2
            ans[i, j] = bits_to_unif(bitsj)

    return ans


# Below are functions to test whether things are going right
# They do not furnish a proof of correctness, but they can catch many errors.
import matplotlib.pyplot as plt


def testrate(mset=range(5, 19), R=50, type="nestu", seed=20210120):
    """
    # R = number of runs
    # mset: plot from 2^5 to 2^19 number of sobol points

    Test function for Sobol sequences.
    Plots empirical variances vs sample size n=2^m.

    # For a smooth function in modest dimensions we should see
    # nearly 1/n^3 variance.
    #
    # This function puts an example plot on screen.
    #
    # This function plots some empirical variances along with
    # reference curves as sample size varies. The selected
    # integrand is smooth but not antithetic or polynomial
    # or spiky or in any way sensitive to base b=2.

    # Test integrand on [0,1]^2; smooth but not specially
    # tuned for Sobol points
    """

    def g(x):
        # x: 1D array of dimension d=2
        d = len(x)
        return np.exp(np.sum(x)) - (np.exp(1) - 1)**d

    def cummean(v):
        return np.cumsum(v) / np.arange(1, len(v) + 1)

    mset = np.array(mset)
    vals = np.zeros((R, len(mset)))
    n_values = 1 << mset  # 2^m for each m in mset

    for r in range(R):
        print(f"Running round {r} of {R}")
        # Generate Sobol points with maximum m
        x = rsobol(m=max(mset), s=2, type=type, seed=seed + r)
        # Apply g row-wise
        y = np.apply_along_axis(g, 1, x)
        # Store cumulative mean at each n=2^m
        for idx, n in enumerate(n_values):
            vals[r, idx] = cummean(y)[:n][-1]

    print("Computing empirical variances...")
    # Compute mean squared across repetitions
    varhat = np.mean(vals**2, axis=0)

    # Plot results
    plt.figure(figsize=(7,5))
    plt.loglog(n_values, varhat, 'o-', label="Estimated variance")
    plt.xlabel("n")
    plt.ylabel("Estimated variance")
    plt.title("Reference curves through first point\nparallel to 1/n^3 and log(n)/n^3")

    # Reference curves
    plt.loglog(n_values, (1/n_values**3) * varhat[0] * n_values[0]**3, 'b-', linewidth=2, label="1/n^3")
    plt.loglog(n_values, (np.log(n_values)/n_values**3) * varhat[0] * n_values[0]**3 / np.log(n_values[0]), 'r-', linewidth=2, label="log(n)/n^3")

    plt.legend()
    plt.savefig(f"testrate_{type}.png")


def canvas():
    """
    Initialize an empty plot with [0,1] x [0,1] axes.
    Aspect ratio is fixed at 1, and no axis labels are shown.
    """
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])


def glines(hn, vn, **kwargs):
    """
    Draw uniform grid lines in the unit square [0,1]^2.

    Parameters:
        hn : int
            Number of horizontal divisions (grid lines along y-axis)
        vn : int
            Number of vertical divisions (grid lines along x-axis)
        kwargs : additional keyword arguments for plt.plot (e.g., color, linestyle)
    """
    # Horizontal lines
    for i in range(hn + 1):
        y = i / hn
        plt.plot([0, 1], [y, y], **kwargs)

    # Vertical lines
    for j in range(vn + 1):
        x = j / vn
        plt.plot([x, x], [0, 1], **kwargs)


def test_grids(filename="rsobolgridtest.pdf", m=8):
    """
    Plot pairs of Sobol sequence dimensions with overlaid uniform grids.

    Parameters:
        filename : str
            PDF file to save the plots.
        m : int
            Log2 of number of points (n = 2^m)
    """
    print(f"Sending test plots to {filename}")

    # Generate Sobol points: 50 dimensions, s=50
    dims = [0, 1, 4, 9, 19, 29, 39, 49]  # zero-indexed in Python
    vals = rsobol(m=m, s=50, seed=1)[:, dims]

    s = vals.shape[1]
    n_points = vals.shape[0]

    # Create a PDF backend
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as pdf:
        for i in range(s - 1):
            for j in range(i + 1, s):
                for k in range(1, m):
                    # Initialize empty unit square
                    plt.figure(figsize=(6, 7))
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xticks([])
                    plt.yticks([])

                    # Plot points
                    plt.scatter(vals[:, i], vals[:, j], s=10, color='black')

                    # Title showing variable pair
                    plt.title(f"Input {dims[j]+1} vs {dims[i]+1}", pad=-10)

                    # Overlay grid lines
                    n1 = 1 << k
                    n2 = 1 << (m - k)
                    glines(n1, n2, color='gray', linestyle='--')

                    pdf.savefig()
                    plt.close()


def verify_lhs(mset=range(3, 13), s=1000, type="nestu", verbose=True):
    """
    Verify that each variable in a Sobol sequence is stratified (Latin hypercube property).

    Parameters:
        mset : iterable of int
            Values of m (log2 of sample size n = 2^m) to test.
        s : int
            Number of dimensions.
        type : str
            Scrambling type: "nestu" or "mato".
        verbose : bool
            Print progress and results.
    """
    allok = True

    for m in mset:
        if verbose:
            print(f"Doing m = {m}")
        n = 1 << m  # 2^m
        vals = rsobol(fn="sobol_Cs.col", m=m, s=s, type=type, seed=1)

        # Scale to integer indices 0..2^m-1
        vals_int = np.floor(n * vals).astype(int)

        # Count unique elements per column
        lenus = np.array([len(np.unique(vals_int[:, j])) for j in range(s)])

        if np.any(lenus != n):
            allok = False
            print(f"LHS problems at m = {m}")
            print("Columns with insufficient unique values:", np.where(lenus != n)[0])

        if verbose and allok:
            print(f"All ok up to m = {m}")

    if allok:
        print("LHS properties look ok")


if __name__ == "__main__":
    testrate(type="hash-owen")
    # res = rsobol(type="hash-owen")
    # print(res)


