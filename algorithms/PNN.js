/**
 * Created by Nguyen Quang Khue on 27-May-17.
 */

/**
 * Node.js module export
 */
if (typeof module !== 'undefined') {
    module.exports.PNN = PNN;
}

/**
 * Input data set of vectors
 * @type {Array}
 */
PNN.prototype.X = [];

/**
 * Input data set of vectors
 * @type {Array}
 */
PNN.prototype.M = Infinity;

/**
 * Ground truth centroid data (if applicable)
 * @type {Array}
 */
PNN.prototype.GT = [];

/**
 * Final solution's codebook
 * @type {Array}
 */
PNN.prototype.centroids = [];

/**
 * Final solution's partition mappings
 * @type {Array}
 */
PNN.prototype.clusterLabels = [];

/**
 * Sum of Squared Error / Total Squared Error score
 * @type {Number}
 */
PNN.prototype.tse = Infinity;

/**
 * Normalised Mean Square Error
 * @type {Number}
 */
PNN.prototype.nmse = Infinity;

/**
 * Centroid Index score
 * @type {Number}
 */
PNN.prototype.ci = Infinity;

/**
 * The iteration where the algorithm stops
 * when no improvements achieved or when
 * the centroids are converged
 * @type {number}
 */
PNN.prototype.stopIter = 0;

/**
 * Algorithm constructor
 * @param X
 * @param M
 * @param GT
 * @constructor
 */
function PNN(X, M, GT) {
    if (!X || M < 0) {
        throw "Invalid parameters";
    }

    this.X = X;
    this.M = M;
    this.N = X.length;

    if (GT && GT.length > 0) {
        this.GT = GT;
    }
}

/**
 * Main execution point
 */
PNN.prototype.execute = function() {
    var initial = this.init();
    var results = this.performPNN(this.X, this.M, initial.codebook, initial.partition);
    var C = results.codebook;
    var P = results.partition;
    var tse = this.sumSquaredError(this.X, C, P);
    this.storeFinalSolution(C, P, tse);
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/
/**
 * Generate initial codebook and partition
 * @returns {{codebook: (Array.<*>|*), partition: Array}}
 */
PNN.prototype.init = function() {
    var C = this.X.clone();
    var P = [];
    for (var i = 0; i < this.N; i++) {
        P[i] = i;
        C[i].size = 1;
    }
    return {
        "codebook": C,
        "partition": P
    }
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/
/**
 * Perform PNN to reduce number of code vectors
 * @param X
 * @param M
 * @param C
 * @param P
 */
PNN.prototype.performPNN = function (X, M, C, P) {
    var Q = [], K = C.length;

    for (var i = 0; i < K; i++) {
        Q[i] = this.findNearestNeighbor(C, i);
    }

    var iterations = 0;
    while (C.length > M) {
        iterations++;
        this.stopIter = iterations;
        var a = this.findMinimumDistance(C, Q);
        var b = Q[a].nearest;
        this.mergeVectors(X, C, P, Q, a, b);
        this.updatePointers(C, Q);
    }

    return {
        "codebook": C,
        "partition": P
    }
};

/**
 * Update pointers
 * @param C
 * @param Q
 */
PNN.prototype.updatePointers = function (C, Q) {
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
PNN.prototype.mergeVectors = function (X, C, P, Q, a, b) {
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
PNN.prototype.fillEmptyPosition = function (C, P, Q, b, last) {
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
PNN.prototype.markClustersForRecalculation = function (C, Q, a, b) {
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
PNN.prototype.createCentroid = function (C1, C2) {
    var n1 = C1.size;
    var n2 = C2.size;
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
PNN.prototype.findMinimumDistance = function (C, Q) {
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
PNN.prototype.findNearestNeighbor = function (C, a) {
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
PNN.prototype.mergeDistortion = function (C1, C2) {
    var n1 = C1.size;
    var n2 = C2.size;
    var factor = (n1 * n2) / (n1 + n2);
    var distance = this.distance(C1, C2, true);
    return factor * distance;
};

/**
 * Join two partitions together
 * @param X
 * @param P
 * @param C
 * @param a
 * @param b
 */
PNN.prototype.joinPartitions = function (P, C, a, b) {
    for (var i = 0; i < this.N; i++) {
        if (P[i] === b) {
            P[i] = a;
        }
    }
    C[a].size = C[a].size + C[b].size;
};

/**
 * Store final solution
 * @param C
 * @param P
 * @param tse
 */
PNN.prototype.storeFinalSolution = function (C, P, tse) {
    this.centroids = C;
    this.clusterLabels = P;
    this.tse = tse;
    this.nmse = this.normalisedMeanSquareError(this.X, C, P, tse);
    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(C, this.GT);
    }
};

/********************************************************************
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/
/**
 * Calculate euclidean distance between two vectors
 * @param x1
 * @param x2
 * @param squared whether we calculate squared distance
 * @returns {*}
 */
PNN.prototype.distance = function (x1, x2, squared) {
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

/**
 * Find nearest vector from a set of vectors
 * @param x
 * @param V
 * @returns {number}
 */
PNN.prototype.findNearestVector = function (x, V) {
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
 * Generate optimal partition
 * @param X
 * @param C
 * @returns {Array}
 */
PNN.prototype.generateOptimalPartition = function (X, C) {
    var P = [];

    for (var i = 0; i < this.N; i++) {
        var j = this.findNearestVector(X[i], C);
        P[i] = j;
        C[j].size = C[j].size + 1;
    }

    return P;
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
PNN.prototype.calculateDissimilarity = function (s1, s2) {

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
PNN.prototype.centroidIndex = function (s1, s2) {

    if (!s1 || !s2 || s1.length <= 0 || s2.length <= 0) {
        throw "Invalid input!";
    }

    var CI1 = this.calculateDissimilarity(s1, s2);
    var CI2 = this.calculateDissimilarity(s2, s1);
    return Math.max(CI1, CI2);
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
PNN.prototype.sumSquaredError = function (X, C, P) {
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
PNN.prototype.normalisedMeanSquareError = function (X, C, P, tse) {

    if (!tse) {
        tse = this.sumSquaredError(X, C, P);
    }

    var n = this.N;
    var d = C[0].length;

    return tse / (n * d);

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
        (this[i] !== value) ? indices.push(i) : this[i];
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
 * Clone an array
 * @returns {Array.<*>}
 */
Array.prototype.clone = function () {
    return this.slice(0, this.length);
};