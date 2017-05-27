/********************************************************************
 * Created by Nguyen Quang Khue on 18-May-17.
 *
 * This file gives implementation of Fast K-Means for clustering,
 * introduced in:
 *
 * "T. Kaukoranta, P. Fränti and O. Nevalainen, "A fast exact GLA
 * based on code vector activity detection", IEEE Trans. on Image
 * Processing, 9 (8), 1337-1342, August 2000"
 *
 * Updated 26-May-2017
 * Nguyen Quang Khue
 * khuenq.devmail@gmail.com / quangn@student.uef.fi
 *
 * ------------------------------------------------------------------
 * INPUT:
 *      X: Data for clustering, represented in the form of 2-D array
 *          * example: var X = [[1,2,4], [1,1,2], [4,5,3], [9,10,0]];
 *      k: The k value. Number of expected code vectors / clusters
 *      T: Maximum number of iterations, set to "Infinity" (without
 *          quotes) to make the algorithm runs until convergence
 *      GT: Groundtruth centroid data, represented in the form of
 *          2-D array, same as input data (if applicable)
 * OUTPUT: (accessed through object properties)
 *      centroids: Final solution's codebook
 *      clusterLabels: Final solution's partition
 *      tse: Final solution's Sum of Squared Error / Total Squared Error
 *      nmse: Final solution's Normalised Mean Square Error
 *      ci: Centroid Index score for evaluate the result validity (optional)
 *          only calculated when ground truth data is supplied
 * -------------------------------------------------------------------
 * [Note]
 * 1. The algorithm is provided as a standalone JavaScript class
 *    with all possible helper functions provided
 * 2. Clustering algorithms are best run with JavaScript engines
 *    such as Node.js
 * -------------------------------------------------------------------
 * USAGE:
 *
 * data = [
 *     [1, 2],
 *     [1.5, 1.8],
 *     [5, 8],
 *     [8, 8],
 *     [1, 0.6],
 *     [9, 11],
 *     [8, 2],
 *     [10, 2],
 *     [9, 3]
 * ];
 *
 * groundTruths = [
 *     [1.1666666666666667, 1.4666666666666666],
 *     [7.333333333333333, 9],
 *     [9, 2.3333333333333335]
 * ];
 *
 * var km = new KMeans(data, 3, Infinity, groundTruths);
 * km.execute();
 *
 * var centroids = km.centroids;
 * var clusterLabels = km.clusterLabels;
 * var sse = km.tse;
 * var ci = km.ci;
 *
 ********************************************************************/

/**
 * Node.js module export
 */
if (typeof module !== 'undefined') {
    module.exports.KMeans = KMeans;
}

/**
 * Input data set of vectors
 * @type {Array}
 */
KMeans.prototype.X = [];

/**
 * The k value, number of code vectors / clusters
 * @type {number}
 */
KMeans.prototype.k = 0;

/**
 * Maximum number of K-Means iterations
 * @type {Number}
 */
KMeans.prototype.T = Infinity;

/**
 * Ground truth centroid data (if applicable)
 * @type {Array}
 */
KMeans.prototype.GT = [];

/**
 * Final solution's codebook
 * @type {Array}
 */
KMeans.prototype.centroids = [];

/**
 * Final solution's partition mappings
 * @type {Array}
 */
KMeans.prototype.clusterLabels = [];

/**
 * Sum of Squared Error / Total Squared Error score
 * @type {Number}
 */
KMeans.prototype.tse = Infinity;

/**
 * Normalised Mean Square Error
 * @type {Number}
 */
KMeans.prototype.nmse = Infinity;

/**
 * Centroid Index score
 * @type {Number}
 */
KMeans.prototype.ci = Infinity;

/**
 * The iteration where the algorithm stops
 * when no improvements achieved or when
 * the centroids are converged
 * @type {number}
 */
KMeans.prototype.stopIter = 0;

/**
 * Constructor of KMeans
 * @param X
 * @param k
 * @param T
 * @param GT
 * @constructor
 */
function KMeans(X, k, T, GT) {

    if (!X || !k) {
        throw "Either input data or number of clusters is not specified!";
    }

    this.X = X;

    // Calculate data size beforehand
    this.N = X.length;
    this.k = k;

    if (T) {
        this.T = T;
    }

    if (GT) {
        this.GT = GT;
    }
}

/**
 * Main execution point
 */
KMeans.prototype.execute = function () {
    var codebook = this.generateRandomCodebook(this.X, this.k);
    var partition = this.generateOptimalPartition(this.X, codebook);
    var results = this.iterateByKMeans(this.X, this.k, codebook, partition);
    this.storeFinalSolution(results);
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/

/**
 * Generate random codebook from a set of data
 * @param X
 * @param M
 * @returns {Array}
 */
KMeans.prototype.generateRandomCodebook = function (X, M) {
    // Get random vector indices
    var indices = Math.randIntList(0, this.N - 1, M, undefined, true);

    // Collect random vectors
    var C = [];
    for (var i = 0; i < indices.length; i++) {
        C[i] = X[indices[i]];
        C[i].size = 0;
    }

    return C;
};

/**
 * Get optimal partition mapping for a certain codebook
 * @param X
 * @param C
 * @returns {Array}
 */
KMeans.prototype.generateOptimalPartition = function (X, C) {
    var P = [];

    for (var i = 0; i < this.N; i++) {
        var j = this.findNearestVector(X[i], C);
        P[i] = j;
        C[j].size = C[j].size + 1;
    }

    return P;
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/

/**
 * Get optimal solution
 * @param X
 * @param M
 * @param C
 * @param P
 * @returns {{codebook: *, partition: *, tse: number}}
 */
KMeans.prototype.iterateByKMeans = function (X, M, C, P) {
    var active = [];
    var changedList = [-1];

    var iterations = 0;

    while (iterations < this.T && changedList.length > 0) {
        var CPrev = C.clone();
        C = this.calculateCentroids(X, C, P);
        this.detectChangedCodeVectors(CPrev, C, active, changedList);
        P = this.reducedSearchPartition(X, C, P, active, changedList);
        iterations++;
        this.stopIter = iterations;
    }

    return {
        "codebook": C,
        "partition": P,
        "tse": this.sumSquaredError(X, C, P)
    }
};

/**
 * Store final results
 * @param results
 */
KMeans.prototype.storeFinalSolution = function (results) {
    this.centroids = results.codebook;
    this.clusterLabels = results.partition;
    this.tse = this.sumSquaredError(this.X, results.codebook, results.partition);
    this.nmse = this.normalisedMeanSquareError(this.X, results.codebook, results.partition, this.tse);

    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(results.codebook, this.GT);
    }
};

/**
 * Detect active code vector (centroids) in the code book
 * and track changes
 * @param CPrev
 * @param CNew
 * @param active
 * @param changedList
 */
KMeans.prototype.detectChangedCodeVectors = function (CPrev, CNew, active, changedList) {
    changedList.length = 0;

    var K = CPrev.length;

    for (var j = 0; j < K; j++) {
        active[j] = false;
        if (!CPrev[j].equals(CNew[j])) {
            changedList.push(j);
            active[j] = true;
        }
    }
};

/**
 * Reduce the search partition by updating cluster labels
 * of each input data vector to the nearest code vector (centroid)
 * @param X
 * @param C
 * @param P
 * @param active
 * @param changedList
 * @returns {*}
 */
KMeans.prototype.reducedSearchPartition = function (X, C, P, active, changedList) {
    var k = 0;

    for (var i = 0; i < this.N; i++) {
        /*var j = P[i];

        if (active[j]) {
            k = this.findNearestVector(X[i], C);
        } else {
            k = this.findNearestInSet(X[i], C, changedList);
        }*/
        k = this.findNearestVector(X[i], C);
        P[i] = k;
        C[k].size += 1;
    }

    return P;
};

/**
 * Find the nearest index of code vector (centroid) that is
 * in the changed list (active code vector)
 * @param vector
 * @param C
 * @param changedList
 * @returns {number}
 */
KMeans.prototype.findNearestInSet = function (vector, C, changedList) {
    var minDist = Infinity;
    var minIdx = 0;
    var len = changedList.length;

    for (var i = 0; i < len; i++) {
        var j = changedList[i];
        var d = this.distance(vector, C[j], true);
        if (d < minDist) {
            minIdx = j;
            minDist = d;
        }
    }

    return minIdx;
};

/**
 * Calculate partition centroids and uses them as code vectors
 * @param X
 * @param C
 * @param P
 * @returns {Array}
 */
KMeans.prototype.calculateCentroids = function (X, C, P) {
    var sum = [], count = [], K = C.length;

    for (var i = 0; i < this.N; i++) {
        var j = P[i];

        if (typeof sum[j] === "undefined") {
            sum[j] = X[i].clone().fill(0);
        }

        sum[j] = sum[j].addArray(X[i]);

        if (typeof count[j] === "undefined") {
            count[j] = 0;
        }

        count[j]++;
    }

    for (var k = 0; k < K; k++) {
        if (typeof sum[k] !== "undefined" && typeof count[k] !== "undefined" && count[k] > 0) {
            C[k] = sum[k].divideBy(count[k]);
        }
        C[k].size = 0;
    }

    return C;
};

/********************************************************************
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/

/**
 * Get the nearest vector to an input vector
 * @param x
 * @param V
 * @returns {number}
 */
KMeans.prototype.findNearestVector = function (x, V) {
    var minDist = Infinity;
    var minIdx = 0;
    var len = V.length;

    for (var i = 0; i < len; i++) {
        var d = this.distance(x, V[i], true);
        if (d < minDist) {
            minDist = d;
            minIdx = i;
        }
    }

    return minIdx;
};

/**
 * Calculate the mean/average vector from a set of vectors
 * @param vectors
 */
KMeans.prototype.calculateMeanVector = function (vectors) {
    var sumVector = vectors[0];
    var nVectors = vectors.length;
    for (var i = 1; i < vectors.length; i++) {
        sumVector = sumVector.addArray(vectors[i]);
    }
    return sumVector.divideBy(nVectors);
};

/**
 * Calculate euclidean distance between two vectors
 * @param x1
 * @param x2
 * @param squared whether we calculate squared distance
 * @returns {*}
 */
KMeans.prototype.distance = function (x1, x2, squared) {
    if (x1.length !== x2.length) {
        throw "Vectors must be of the same length!";
    }

    // Initialize distance variable
    var d = 0;

    var n = x1.length;

    // Calculate distance between each feature
    for (var i = 0; i < n; i++) {
        d += Math.pow(x1[i] - x2[i], 2);
    }

    if (squared) {
        return d;
    }

    return Math.sqrt(d);
};

/********************************************************************
 * Objective function                                               *
 ********************************************************************/

/**
 * Calculate Sum of Squared Error / Total Squared Error
 * @param X
 * @param C
 * @param P
 * @returns {number}
 */
KMeans.prototype.sumSquaredError = function (X, C, P) {
    var tse = 0;
    for (var i = 0; i < this.N; i++) {
        var j = P[i];
        if (C[j]) {
            tse += this.distance(X[i], C[j], true);
        }
    }
    return tse;
};

/**
 * Calculate Normalised Mean Square Error
 * @param X
 * @param C
 * @param P
 * @param tse
 * @returns {number}
 */
KMeans.prototype.normalisedMeanSquareError = function (X, C, P, tse) {

    if (!tse) {
        tse = this.sumSquaredError(X, C, P);
    }

    var n = this.N;
    var d = C[0].length;

    return tse / (n * d);

};

/********************************************************************
 * Centroid index calculation                                       *
 ********************************************************************/

/**
 * Calculate one-way dissimilarity score between
 * two sets of data
 * @param s1
 * @param s2
 * @returns {number}
 * @constructor
 */
KMeans.prototype.calculateDissimilarity = function (s1, s2) {

    // Initialization
    var k1 = s1.length;
    var k2 = s2.length;
    var q = []; // Nearest neighbor mappings
    var orphans = []; // A set of orphans

    // Map each of c1 with its nearest centroid from c2
    for (var i = 0; i < k1; i++) {
        q[i] = this.findNearestVector(s1[i], s2);
    }

    for (var k = 0; k < k2; k++) {
        // Get the number of all mappings that does not belong to current label
        var sq = q.allIndexNotOf(k).length;

        if (sq === k1) {
            orphans[k] = 1
        } else {
            orphans[k] = 0;
        }
    }

    return orphans.sum();

};

/**
 * Calculate centroid index given two sets of centroid data
 * @param s1
 * @param s2
 * @returns {number}
 */
KMeans.prototype.centroidIndex = function (s1, s2) {

    if (!s1 || !s2 || s1.length <= 0 || s2.length <= 0) {
        throw "Invalid input!";
    }

    var CI1 = this.calculateDissimilarity(s1, s2);
    var CI2 = this.calculateDissimilarity(s2, s1);
    return Math.max(CI1, CI2);
};

/********************************************************************
 * SUPPORT FUNCTIONS ADDED TO JAVASCRIPT ARRAY OBJECT LIBRARY       *
 ********************************************************************/

/**
 * Find all indices of an element in an array
 * @param value
 */
Array.prototype.allIndexOf = function (value) {
    var indices = [];
    for (var i = 0; i < this.length; i++) {
        (this[i] === value) ? indices.push(i) : this[i];
    }
    return indices;
};

/**
 * Get all indices of elements that do not equal to a specific value
 * @param value
 * @returns {Array}
 */
Array.prototype.allIndexNotOf = function (value) {
    var indices = [];
    for (var i = 0; i < this.length; i++) {
        if (this[i] !== value) {
            indices.push(i);
        }
    }
    return indices;
};

/**
 * Get elements of an array based on a list of indices
 * @param indices
 * @returns {Array}
 */
Array.prototype.getElementsByIndices = function (indices) {
    var elements = [];
    for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        elements[i] = this[idx].slice(0, this[idx].length);
    }
    return elements;
};

/**
 * Add elements of another array of the same length
 * to current array one-by-one
 * @param arr
 */
Array.prototype.addArray = function (arr) {

    var len = this.length;

    if (len !== arr.length) {
        throw "Input array must have the same length with current array!";
    }

    for (var i = 0; i < len; i++) {
        this[i] = this[i] + arr[i];
    }

    return this;
};

/**
 * Divide elements of current array by a number
 * @param val
 */
Array.prototype.divideBy = function (val) {

    // Convert value to floating points for calculation
    val = parseFloat(val);

    if (!val || typeof(val) !== "number") {
        throw "Division by zero or invalid value";
    }

    var len = this.length;

    for (var i = 0; i < len; i++) {
        this[i] /= val;
    }

    return this;
};

/**
 * Divide elements of current array by a number
 * @param val
 */
Array.prototype.multiplyBy = function (val) {

    // Convert value to floating points for calculation
    val = parseFloat(val);

    if (typeof(val) !== "number") {
        throw "Invalid value";
    }

    var len = this.length;

    for (var i = 0; i < len; i++) {
        this[i] *= val;
    }

    return this;
};

/**
 * Check whether array contains an element
 * @param element
 * @returns {boolean}
 */
Array.prototype.hasElement = function (element) {

    for (var i = 0; i < this.length; i++) {
        if (this[i] instanceof Array) {
            if (this[i].equals(element)) {
                return true;
            }
        } else {
            if (this[i] === element) {
                return true;
            }
        }
    }

    return false;
};

/**
 * Get the sum of all array elements
 * @returns {number}
 */
Array.prototype.sum = function () {

    var n = this.length, s = 0;

    for (var i = 0; i < n; i++) {
        s += this[i];
    }

    return s;
};

/**
 * Function to compare between arrays
 * @param array
 * @param strict
 * @returns {boolean}
 */
Array.prototype.equals = function (array, strict) {
    if (!array)
        return false;

    if (arguments.length === 1)
        strict = true;

    if (this.length !== array.length)
        return false;

    for (var i = 0; i < this.length; i++) {
        if (this[i] instanceof Array && array[i] instanceof Array) {
            if (!this[i].equals(array[i], strict))
                return false;
        } else if (strict && this[i] !== array[i]) {
            return false;
        } else if (!strict) {
            return this.sort().equals(array.sort(), true);
        }
    }
    return true;
};

/**
 * Clone an array
 * @returns {Array.<*>}
 */
Array.prototype.clone = function () {
    return this.slice(0, this.length);
};


/********************************************************************
 * SUPPORT FUNCTIONS ADDED TO JAVASCRIPT MATHEMATICS LIBRARY        *
 ********************************************************************/

/**
 * Get a random integer between a range
 * @param min
 * @param max
 * @param exclude ignore if the result is this value
 * @returns {*}
 */
Math.randInt = function (min, max, exclude) {
    if (exclude === null || exclude === undefined || exclude === false) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    var result = exclude;
    while (result === exclude) {
        result = Math.floor(Math.random() * (max - min + 1)) + min;
    }
    return result;
};

/**
 * Get a list of random integers
 * @param min
 * @param max
 * @param length
 * @param exclude ignore if the result is this value
 * @param unique whether the list of random numbers should contain only unique numbers
 * @returns {Array}
 */
Math.randIntList = function (min, max, length, exclude, unique) {
    var rand = [];
    while (rand.length < length) {
        var randNum = Math.randInt(min, max, exclude);
        if (unique) {
            if (!rand.hasElement(randNum)) {
                rand.push(randNum);
            }
        } else {
            rand.push(randNum);
        }
    }
    return rand;
};