/********************************************************************
 * Created by Nguyen Quang Khue on 18-May-17.
 *
 * This file gives implementation of Mean Shift algorithm for clustering,
 * originally presented in:
 *
 * " Y. Cheng, "Mean shift, mode seeking, and clustering", IEEE Trans.
 * on Pattern analysis and Machine Intelligence, 17 (8), 790-799, 1995."
 *
 * Updated 26-May-2017
 * Nguyen Quang Khue
 * khuenq.devmail@gmail.com / quangn@student.uef.fi
 *
 * ------------------------------------------------------------------
 * INPUT:
 *      X: Data for clustering, represented in the form of 2-D array
 *          * example: var X = [[1,2,4], [1,1,2], [4,5,3], [9,10,0]];
 *      R: Kernel radius for gathering vectors to perform mean shifting
 *      M: Number of expected code vectors / clusters
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
 * 3. This implementation uses flat kernel for performing the mean
 *    shifting process
 * 4. References to algorithms used for optimize/fine-tuning clustering
 *    results:
 *    a. Fast K-Means:
 *      "T. Kaukoranta, P. Fränti and O. Nevalainen, "A fast exact GLA
 *      based on code vector activity detection", IEEE Trans. on Image
 *      Processing, 9 (8), 1337-1342, August 2000"
 *    b. Fast exact PNN:
 *      "Franti, P., Kaukoranta, T., 1998. Fast implementation of the
 *      optimal PNN method. In: IEEE Proceedings of the
 *      International Conference on Image Processing (ICIPÕ98),
 *      Chicago, Illinois, USA (revised version will appear in IEEE
 *      Transactions on Image Processing)."
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
 * var ms = new MeanShift(data, 3, 4, 20, groundTruths);
 * ms.execute();
 *
 * var centroids = ms.centroids;
 * var clusterLabels = ms.clusterLabels;
 * var sse = ms.tse;
 * var ci = ms.ci;
 *
 ********************************************************************/

/**
 * Node.js module export
 */
if (typeof module !== 'undefined') {
    module.exports.MeanShift = MeanShift;
}

/**
 * Input data set of vectors
 * @type {Array}
 */
MeanShift.prototype.X = [];

/**
 * Kernel radius
 * @type {number}
 */
MeanShift.prototype.R = Infinity;

/**
 * Desired number of code vectors / clusters
 * @type {number}
 */
MeanShift.prototype.M = Infinity;

/**
 * Ground truth centroid data (if applicable)
 * @type {Array}
 */
MeanShift.prototype.GT = [];

/**
 * Final solution's codebook
 * @type {Array}
 */
MeanShift.prototype.centroids = [];

/**
 * Final solution's partition mappings
 * @type {Array}
 */
MeanShift.prototype.clusterLabels = [];

/**
 * Sum of Squared Error / Total Squared Error
 * @type {Number}
 */
MeanShift.prototype.tse = Infinity;

/**
 * Normalised Mean Square Error
 * @type {Number}
 */
MeanShift.prototype.nmse = Infinity;

/**
 * Centroid Index score
 * @type {Number}
 */
MeanShift.prototype.ci = Infinity;

/**
 * The iteration where the algorithm stops
 * when no improvements achieved or when
 * the centroids are converged
 * @type {number}
 */
MeanShift.prototype.stopIter = 0;

/**
 * Max number of iterations, prevent no convergence
 * when determine radius automatically
 * @type {number}
 */
MeanShift.prototype.maxIter = 50;

/**
 * Constructor of the algorithm class
 * @param X Input data set of vectors
 * @param R Kernel radius
 * @param M Desired number of code vectors / clusters
 * @param GT Ground truth centroids data (if applicable)
 * @constructor
 */
function MeanShift(X, R, M, GT) {
    if (!X || R < 0) {
        throw "Invalid input parameters";
    }

    this.X = X;
    this.R = R * R; // Squared radius
    this.N = X.length;

    if (M && M > 0 && M !== Infinity) {
        this.M = M
    }

    if (GT && GT.length > 0) {
        this.GT = GT;
    }
}

/**
 * Main execution point
 */
MeanShift.prototype.execute = function () {
    if (!this.R || this.R === Infinity || this.R <= 0) {
        this.R = this.determineKernelRadius(this.X);
    }

    var C = this.getOptimalCodebook(this.X, this.R);
    var P = this.getOptimalPartition(this.X, C);

    var tuned = this.tuningSolution(this.X, this.M, C, P);
    C = tuned.codebook;
    P = tuned.partition;

    var tse = this.sumSquaredError(this.X, C, P);
    this.storeFinalSolution(C, P, tse);
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/

/**
 * Get all vectors as initial codebook
 * @param X
 * @returns {Array.<*>|*}
 */
MeanShift.prototype.getInitialCodebook = function (X) {
    return X.clone();
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/
/**
 * Heuristic method for determine kernel radius
 * @param X
 * @returns {number}
 */
MeanShift.prototype.determineKernelRadius = function (X) {
    var N = this.N;
    var D = X[0].length;
    var k = Math.floor(Math.sqrt(N) / D) + D;

    var cb = this.generateRandomCodebook(X, k);
    var pt = this.getOptimalPartition(X, cb);

    var totalAvgDist = 0;
    for (var i = 0; i < this.N; i++) {
        var j = pt[i];
        var d = this.distance(X[i], cb[j], true);
        if (cb[j].size > 0) {
            totalAvgDist += d / cb[j].size;
        }
    }

    return Math.round(totalAvgDist / k);
};

/**
 * Get optimal codebook iteratively
 * @param X
 * @param R
 * @returns {Array.<*>|*}
 */
MeanShift.prototype.getOptimalCodebook = function (X, R) {
    var C = this.getInitialCodebook(X);
    var optimize = false, iterations = 0;

    while (!optimize && iterations < this.maxIter) {
        iterations++;
        this.stopIter = iterations;

        var CPrev = C.clone();
        C = this.updateCentroids(X, C, R);
        if (C.equals(CPrev)) {
            optimize = true;
        }
    }

    return C;
};

/**
 * Get optimal partition from codebook
 * @param X
 * @param C
 * @returns {Array}
 */
MeanShift.prototype.getOptimalPartition = function (X, C) {
    var P = [];
    for (var i = 0; i < this.N; i++) {
        var j = this.findNearestVector(X[i], C);
        P[i] = j;
        C[j].size += 1;
    }
    return P;
};

/**
 * Store the final solution
 * @param C
 * @param P
 * @param tse
 */
MeanShift.prototype.storeFinalSolution = function (C, P, tse) {
    this.centroids = C;
    this.clusterLabels = P;
    this.tse = tse;
    this.nmse = this.normalisedMeanSquareError(this.X, C, P, tse);
    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(C, this.GT);
    }
};

/**
 * Update centroids
 * @param X
 * @param C
 * @param R
 * @returns {Array}
 */
MeanShift.prototype.updateCentroids = function (X, C, R) {
    var CNew = [], len = C.length;

    for (var i = 0; i < len; i++) {

        var withinRadius = [];

        for (var j = 0; j < this.N; j++) {
            // Calculate distance between current code vector and all vectors in the data set
            var d = this.distance(C[i], X[j], true);

            // Search for vectors within kernel radius
            if (d <= R) {
                withinRadius.push(X[j]);
            }
        }

        if (withinRadius.length > 0) {
            // Push the mean vector of such vectors to the new codebook
            var centroid = this.getMeanVector(withinRadius);
            centroid.size = 0;
            CNew.push(centroid);
        }

    }

    // Remove duplicate centroids
    CNew = this.sortCodebook(this.filterDuplicateCentroids(CNew));

    return CNew;
};

/********************************************************************
 * TUNING ROUTINES                                                    *
 ********************************************************************/
/**
 * Fine-tune the final solution
 * @param X
 * @param M
 * @param C
 * @param P
 * @returns {{codebook: *, partition: *}}
 */
MeanShift.prototype.tuningSolution = function(X, M, C, P) {
    if (typeof M !== "undefined" && M !== Infinity && M > 0) {

        if (C.length > M) {
            this.performPNN(X, M, C, P);
        } else if (C.length < M) {
            C = this.splitCentroids(X, M, C);
            P = this.getOptimalPartition(X, C);
        }

    } else {
        C = this.removeLowDensityClusters(C);
        P = this.getOptimalPartition(X, C);
    }

    return {
        "codebook": C,
        "partition": P
    }
};

/**
 * Heuristic centroid splitting method
 * @param X
 * @param M
 * @param C
 * @returns {*}
 */
MeanShift.prototype.splitCentroids = function (X, M, C) {
    for (var i = 0; i < C.length; i++) {
        if (C.length === M) {
            return C;
        }
        var c = C[i].clone();

        // Choose two furthest vectors
        var v1 = X[this.getFurthestVector(c, X)];
        var v2 = X[this.getFurthestVector(v1, X)];

        // Get mean vector as new code vector
        var v3 = this.getMeanVector([c, v1]);
        var v4 = this.getMeanVector([c, v2]);
        v3.size = 0;
        v4.size = 0;

        if (C.length < M) {
            C.push(v3);
        }

        if (C.length < M) {
            C.push(v4);
        }
    }
    return C;
};

/**
 * Get two furthest vectors
 * @param x
 * @param X
 * @returns {number}
 */
MeanShift.prototype.getFurthestVector = function (x, X) {
    var maxDist = 0;
    var maxIdx = 0;

    for (var i = 0; i < this.N; i++) {
        var d = this.distance(x, X[i], true);
        if (d > maxDist) {
            maxDist = d;
            maxIdx = i;
        }
    }

    return maxIdx;
};

/**
 * Remove low density clusters
 * @param C
 * @returns Array
 */
MeanShift.prototype.removeLowDensityClusters = function (C) {
    var sizes = 0, len = C.length;

    // Get density value by calculate average number of vectors
    // in each cluster
    for (var j = 0; j < len; j++) {
        sizes += C[j].size;
    }
    var density = Math.ceil(sizes / len);

    // Keep only centroids that have high cluster density
    var newCodebook = [];
    for (var k = 0; k < len; k++) {
        if (C[k].size >= density) {
            var centroid = C[k].clone();
            centroid.size = 0;
            newCodebook.push(centroid);
        }
    }

    return newCodebook;
};

/********************************************************************
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/

/**
 * Determine whether two codebooks are the same
 * @param C1
 * @param C2
 * @returns {boolean}
 */
MeanShift.prototype.areSameCodebooks = function (C1, C2) {
    var K = C1.length;
    for (var i = 0; i < K; i++) {
        if (!C1[i].equals(C2[i])) {
            return false;
        }
    }
    return true;
};

/**
 * Generate random codebook from a set of data
 * @param X
 * @param M
 * @returns {Array}
 */
MeanShift.prototype.generateRandomCodebook = function (X, M) {
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
 * Sort the codebook by its code vectors' first dimension
 * @param C
 * @returns {Array.<T>|*}
 */
MeanShift.prototype.sortCodebook = function (C) {
    return C.sort(function (a, b) {
        return a[0] - b[0];
    });
};

/**
 * Remove duplicated centroids from the codebook
 * @param C
 * @returns {Array}
 */
MeanShift.prototype.filterDuplicateCentroids = function (C) {
    var seen = {};
    var out = [];
    var len = C.length;
    var j = 0;
    for (var i = 0; i < len; i++) {
        var item = C[i];
        if (seen[item] !== 1) {
            seen[item] = 1;
            out[j++] = item;
        }
    }
    return out;
};

/**
 * Get mean vector
 * @param V
 */
MeanShift.prototype.getMeanVector = function (V) {
    var sum = V[0].clone(); // Set initial sum to first vector
    var nVecs = V.length;

    // Start from the second vector
    for (var i = 1; i < nVecs; i++) {
        sum.addArray(V[i]);
    }

    return sum.divideBy(nVecs);
};

/**
 * Get the nearest vector to an input vector
 * @param x
 * @param vectors
 * @returns {number}
 */
MeanShift.prototype.findNearestVector = function (x, vectors) {

    var minDist = Infinity;
    var minIdx = 0;

    for (var i = 0; i < vectors.length; i++) {
        var d = this.distance(x, vectors[i], true);
        if (d < minDist) {
            minDist = d;
            minIdx = i;
        }
    }

    return minIdx;
};

/**
 * Calculate euclidean distance between two vectors
 * @param x1
 * @param x2
 * @param squared whether we calculate squared distance
 * @returns {*}
 */
MeanShift.prototype.distance = function (x1, x2, squared) {
    if (x1.length !== x2.length) {
        console.trace();
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
 * PNN ROUTINES                                                     *
 ********************************************************************/
/**
 * Perform PNN to reduce number of code vectors
 * @param X
 * @param M
 * @param C
 * @param P
 */
MeanShift.prototype.performPNN = function (X, M, C, P) {
    var Q = [], K = C.length;

    for (var i = 0; i < K; i++) {
        Q[i] = this.findNearestNeighbor(C, i);
    }

    while (C.length > M) {
        var a = this.findMinimumDistance(C, Q);
        var b = Q[a].nearest;
        this.mergeVectors(X, C, P, Q, a, b);
        this.updatePointers(C, Q);
    }
};

/**
 * Update pointers
 * @param C
 * @param Q
 */
MeanShift.prototype.updatePointers = function (C, Q) {
    var K = C.length;
    for (var i = 0; i < K; i++) {
        if (Q[i].recalculate) {
            Q[i] = this.findNearestNeighbor(C, i);
            Q[i].recalculate = true;
        }
    }
};

/**
 * Merge two vectors together
 * @param X
 * @param C
 * @param P
 * @param Q
 * @param a
 * @param b
 */
MeanShift.prototype.mergeVectors = function (X, C, P, Q, a, b) {
    // Swap
    if (a > b) {
        var tmp = a;
        a = b;
        b = tmp;
    }
    var last = C.length - 1;
    this.markClustersForRecalculation(C, Q, a, b);
    var tmpSize = C[a].size;
    C[a] = this.createCentroid(C[a], C[b]);
    C[a].size = tmpSize;
    this.joinPartitions(P, C, a, b);
    this.fillEmptyPosition(C, P, Q, b, last);
    C.length--;
};

/**
 * Fill empty positions
 * @param C
 * @param P
 * @param Q
 * @param b
 * @param last
 */
MeanShift.prototype.fillEmptyPosition = function (C, P, Q, b, last) {
    if (b !== last) {
        C[b] = C[last].clone();
        C[b].size = C[last].size;

        for (var j = 0; j < this.N; j++) {
            if (P[j] === last) {
                P[j] = b;
            }
        }

        Q[b] = {
            "nearest": Q[last].nearest,
            "distance": Q[last].distance,
            "recalculate": Q[last].recalculate
        };

        var K = C.length;
        for (var i = 0; i < K; i++) {
            if (Q[i].nearest === last) {
                Q[i].nearest = b;
            }
        }
    }
};

/**
 * Mark clusters for recalculation
 * @param C
 * @param Q
 * @param a
 * @param b
 */
MeanShift.prototype.markClustersForRecalculation = function (C, Q, a, b) {
    var len = C.length;
    for (var i = 0; i < len; i++) {
        Q[i].recalculate = (Q[i].nearest === a || Q[i].nearest === b);
    }
};

/**
 * Create new weighted centroids from two centroids
 * @param C1
 * @param C2
 */
MeanShift.prototype.createCentroid = function (C1, C2) {
    var n1 = C1.size + 1;
    var n2 = C2.size + 1;
    var C1New = C1.clone().multiplyBy(n1);
    var C2New = C2.clone().multiplyBy(n2);
    return C1New.addArray(C2New).divideBy(n1 + n2);
};

/**
 * Find minimum distance inside distance mapping
 * @param C
 * @param Q
 * @returns {number}
 */
MeanShift.prototype.findMinimumDistance = function (C, Q) {
    var minDist = Infinity;
    var minIdx = 0;
    var K = C.length;
    for (var i = 0; i < K; i++) {
        if (Q[i].distance < minDist) {
            minIdx = i;
            minDist = Q[i].distance
        }
    }
    return minIdx;
};

/**
 * Find nearest neighbor of vector a
 * @param C
 * @param a
 * @returns {{nearest: number, distance: Number, recalculate: boolean}}
 */
MeanShift.prototype.findNearestNeighbor = function (C, a) {
    var q = {
        "nearest": 0,
        "distance": Infinity,
        "recalculate": false
    };
    var K = C.length;
    for (var i = 0; i < K; i++) {
        var d = this.mergeDistortion(C[a], C[i]);
        if (a !== i && d < q.distance) {
            q.nearest = i;
            q.distance = d;
        }
    }
    return q;
};

/**
 * Calculate merge distortion between two code vectors
 * @param C1
 * @param C2
 * @returns {number}
 */
MeanShift.prototype.mergeDistortion = function (C1, C2) {
    var n1 = C1.size + 1;
    var n2 = C2.size + 1;
    var factor = (n1 * n2) / (n1 + n2);
    var distance = this.distance(C1, C2, true);
    return factor * distance;
};

/**
 * Join two partitions together
 * @param P
 * @param C
 * @param a
 * @param b
 */
MeanShift.prototype.joinPartitions = function (P, C, a, b) {
    for (var i = 0; i < this.N; i++) {
        if (P[i] === b) {
            P[i] = a;
        }
    }
    C[a].size = C[a].size + C[b].size;
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
MeanShift.prototype.sumSquaredError = function (X, C, P) {
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
MeanShift.prototype.normalisedMeanSquareError = function (X, C, P, tse) {

    if (!tse) {
        tse = this.sumSquaredError(C, P);
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
MeanShift.prototype.calculateDissimilarity = function (s1, s2) {

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
MeanShift.prototype.centroidIndex = function (s1, s2) {

    if (!s1 || !s2 || s1.length <= 0 || s2.length <= 0) {
        throw "Invalid input!";
    }

    var CI1 = this.calculateDissimilarity(s1, s2);
    var CI2 = this.calculateDissimilarity(s2, s1);
    return Math.max(CI1, CI2);
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
            if (rand.indexOf(randNum) === -1) {
                rand.push(randNum);
            }
        } else {
            rand.push(randNum);
        }
    }
    return rand;
};