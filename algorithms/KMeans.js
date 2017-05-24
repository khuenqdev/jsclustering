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
 * Updated 18-May-2017
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
    var active = [];
    var changedList = [-1];

    var codebook = this.generateRandomCodebook(this.X, this.k);
    var partition = this.getOptimalPartition(this.X, codebook);
    var iterations = 0;

    while (iterations < this.T && changedList.length > 0) {
        var prevCodebook = codebook.slice(0, codebook.length);
        codebook = this.calculateCentroids(this.X, prevCodebook, partition);
        var changes = this.detectChangedCodeVectors(prevCodebook, codebook, active, changedList);
        changedList = changes.changedList;
        active = changes.activeList;
        partition = this.reducedSearchPartition(this.X, codebook, partition, active, changedList);
        this.stopIter = iterations + 1;
        iterations++;
    }

    this.centroids = codebook;
    this.clusterLabels = partition;
    this.tse = this.sumSquaredError(codebook, partition);
    this.nmse = this.normalisedMeanSquareError(codebook, partition, this.tse);

    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(codebook, this.GT);
    }
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/

/**
 * Generate random codebook from a set of data
 * @param data
 * @param k
 * @returns {Array}
 */
KMeans.prototype.generateRandomCodebook = function (data, k) {
    var indices = Math.randIntList(0, data.length - 1, k, undefined, true);
    return data.getElementsByIndices(indices);
};

/**
 * Get optimal partition mapping for a certain codebook
 * @param data
 * @param codebook
 * @returns {Array}
 */
KMeans.prototype.getOptimalPartition = function (data, codebook) {
    var partition = [];
    for (var i = 0; i < data.length; i++) {
        partition[i] = this.findNearestVector(data[i], codebook);
    }
    return partition;
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/

/**
 * Detect active code vector (centroids) in the code book
 * and track changes
 * @param prevCodebook
 * @param newCodebook
 * @param active
 * @param changedList
 */
KMeans.prototype.detectChangedCodeVectors = function (prevCodebook, newCodebook, active, changedList) {

    changedList.length = 0; // Make the changed list empty

    // For each code vector of the previous code book
    for (var j = 0; j < prevCodebook.length; j++) {

        active[j] = false;

        // If the previous code vector and the new code vector are not the same centroid
        if (!newCodebook[j].equals(prevCodebook[j])) {
            if (!changedList.hasElement(j)) {
                changedList.push(j); // Put the changed code vector index to the changed list
            }
            active[j] = true; // Mark it as active
        }

    }

    return {
        "changedList": changedList,
        "activeList": active
    };

};

/**
 * Reduce the search partition by updating cluster labels
 * of each input data vector to the nearest code vector (centroid)
 * @param data
 * @param codebook
 * @param partition
 * @param active
 * @param changedList
 * @returns {*}
 */
KMeans.prototype.reducedSearchPartition = function (data, codebook, partition, active, changedList) {

    // For each input data vector
    for (var i = 0; i < data.length; i++) {

        if (changedList.length > 1) {
            var j = partition[i]; // Get its current cluster label in the partition mapping

            if (active[j]) { // If the code vector corresponding to the cluster is active
                // Find and assign the current vector to the cluster of the nearest code vector
                partition[i] = this.findNearestVector(data[i], codebook);
            } else {
                // Otherwise, find and assign the current vector to the cluster of the nearest code vector in the active code vector list
                partition[i] = this.findNearestCentroidInChangedList(data[i], codebook, changedList);
            }
        } else {
            partition[i] = this.findNearestVector(data[i], codebook);
        }

    }

    return partition;

};

/**
 * Find the nearest index of code vector (centroid) that is
 * in the changed list (active code vector)
 * @param vector
 * @param codebook
 * @param changedList
 * @returns {number}
 */
KMeans.prototype.findNearestCentroidInChangedList = function (vector, codebook, changedList) {

    var minDist = Infinity;
    var minIndex = 0;

    for (var i = 0; i < changedList.length; i++) {
        var j = changedList[i];
        var d = this.distance(vector, codebook[j], true);
        if (d < minDist) {
            minIndex = j;
            minDist = d;
        }
    }

    return minIndex;

};

/**
 * Calculate partition centroids and uses them as code vectors
 * @param data
 * @param codebook
 * @param partition
 * @returns {Array}
 */
KMeans.prototype.calculateCentroids = function (data, codebook, partition) {
    var newCodebook = [];

    for (var i = 0; i < codebook.length; i++) {

        var indices = partition.allIndexOf(i);
        var vectors = data.getElementsByIndices(indices);

        // Default to old centroid
        var centroid = codebook[i];

        if (vectors.length > 0) { // If the list of vectors is not empty
            centroid = this.calculateMeanVector(vectors);
        }

        newCodebook[i] = centroid;
    }

    return newCodebook;
};

/********************************************************************
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/

/**
 * Get the nearest vector to an input vector
 * @param x
 * @param vectors
 * @returns {number}
 */
KMeans.prototype.findNearestVector = function (x, vectors) {

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
 * Calculate the mean/average vector from a set of vectors
 * @param vectors
 */
KMeans.prototype.calculateMeanVector = function (vectors) {
    var sumVector = vectors[0].slice(0, vectors[0].length);
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
 * @param codebook
 * @param partition
 * @returns {number}
 */
KMeans.prototype.sumSquaredError = function (codebook, partition) {
    var tse = 0;
    for (var i = 0; i < this.N; i++) {
        var j = partition[i];
        if (codebook[j]) {
            tse += this.distance(this.X[i], codebook[j], true);
        }
    }
    return tse;
};

/**
 * Calculate Normalised Mean Square Error
 * @param codebook
 * @param partition
 * @param tse
 * @returns {number}
 */
KMeans.prototype.normalisedMeanSquareError = function (codebook, partition, tse) {

    if (!tse) {
        tse = this.sumSquaredError(codebook, partition);
    }

    var n = this.N;
    var d = codebook[0].length;

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
 * Count how many times a value occurs in
 * an array
 * @param x
 * @returns {Number}
 */
Array.prototype.countVal = function (x) {
    var el = this.filter(function (val) {
        return val === x;
    });
    return el.length;
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