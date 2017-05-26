/********************************************************************
 * Created by Nguyen Quang Khue on 17-May-17.
 *
 * This file gives implementation of Genetic Algorithm for clustering,
 * originally presented in:
 *
 * "P. Fränti, "Genetic algorithm with deterministic
 * crossover for vector quantization", Pattern Recognition
 * Letters, 21 (1), 61-68, 2000"
 *
 * Updated 26-May-2017
 * Nguyen Quang Khue
 * khuenq.devmail@gmail.com / quangn@student.uef.fi
 *
 * ------------------------------------------------------------------
 * INPUT:
 *      X: Data for clustering, represented in the form of 2-D array
 *          * example: var X = [[1,2,4], [1,1,2], [4,5,3], [9,10,0]];
 *      M: Number of expected code vectors / clusters
 *      S: Solution population size
 *      T: Number of iterations (maximum number of solution generations)
 *      GT: Groundtruth centroid data, represented in the form of
 *          2-D array, same as input data (if applicable)
 * OUTPUT: (accessed through object properties)
 *      centroids: Best solution's codebook
 *      clusterLabels: Best solution's partition
 *      tse: Best solution's Sum of Squared Error / Total Squared Error
 *      nmse: Best solution's Normalised Mean Square Error
 *      ci: Centroid Index score for evaluate the result validity (optional)
 *          only calculated when ground truth data is supplied
 * -------------------------------------------------------------------
 * [Note]
 * 1. The algorithm is provided as a standalone JavaScript class
 *    with all possible helper functions provided
 * 2. Clustering algorithms are best run with JavaScript engines
 *    such as Node.js
 * 3. References to algorithms used for optimize/fine-tuning clustering
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
 * var ga = new GeneticAlgorithm(data, 3, 4, 20, groundTruths);
 * ga.execute();
 *
 * var centroids = ga.centroids;
 * var clusterLabels = ga.clusterLabels;
 * var sse = ga.tse;
 * var ci = ga.ci;
 *
 ********************************************************************/

/**
 * Node.js module export
 */
if (typeof module !== 'undefined') {
    module.exports.GeneticAlgorithm = GeneticAlgorithm;
}

/**
 * Data set of vectors, represented in the form
 * of 2-D array
 * @type {Array}
 */
GeneticAlgorithm.prototype.X = [];

/**
 * Desired number of code vectors / clusters
 * @type {number}
 */
GeneticAlgorithm.prototype.M = Infinity;

/**
 * Solution population size
 * @type {number}
 */
GeneticAlgorithm.prototype.S = 45;

/**
 * Number of iterations (maximum number of generations)
 * @type {number}
 */
GeneticAlgorithm.prototype.T = 50;

/**
 * Ground truth data contains centroid vectors (if applicable)
 * @type {Array}
 */
GeneticAlgorithm.prototype.GT = [];

/**
 * Number of K-Means iterations
 * @type {number}
 */
GeneticAlgorithm.prototype.maxKMeansIterations = 2;

/**
 * Size of the cross set
 * @type {number}
 */
GeneticAlgorithm.prototype.crossSetSize = 0;

/**
 * Solution population
 * @type {Array}
 */
GeneticAlgorithm.prototype.solutions = [];

/**
 * Best codebook
 * @type {Array}
 */
GeneticAlgorithm.prototype.centroids = [];

/**
 * Best partition
 * @type {Array}
 */
GeneticAlgorithm.prototype.clusterLabels = [];

/**
 * Iteration when the algorithm stop
 * @type {Number}
 */
GeneticAlgorithm.prototype.stopIter = Infinity;

/**
 * Normalised Mean Squared Error
 * @type {Number}
 */
GeneticAlgorithm.prototype.nmse = Infinity;

/**
 * Centroid Index (if applicable)
 * @type {Number}
 */
GeneticAlgorithm.prototype.ci = Infinity;

/**
 * Total Squared Error / Sum Squared Error
 * @type {Number}
 */
GeneticAlgorithm.prototype.tse = Infinity;

/**
 * Constructor
 * @param X Input data
 * @param M Number of code vectors / clusters
 * @param S Population size
 * @param T Max iteration
 * @param GT Ground truth codebook (if applicable)
 * @constructor
 */
function GeneticAlgorithm(X, M, S, T, GT) {
    if (!X || !M || !S || !T) {
        throw "Invalid parameters";
    }

    this.X = X;
    this.M = M;
    this.S = S;
    this.T = T;
    this.N = X.length; // Sample size

    if (typeof GT !== "undefined") {
        this.GT = GT;
    }
}

/**
 * Main execution point
 */
GeneticAlgorithm.prototype.execute = function () {
    // Algorithm initialization
    this.init();

    // Main iterations
    for (var i = 0; i < this.T; i++) {
        this.stopIter = i + 1;

        // Generate new solutions
        this.solutions = this.generateNewSolutions(this.X, this.M, this.S, this.solutions);

        // Sort the solution by TSE score from low to high
        this.sortSolutions(this.solutions);

        // Get the best solution
        var bestSolution = this.solutions[0];

        // Store best solution
        if (bestSolution.tse < this.tse) {
            this.storeBestSolution(bestSolution);
        } else if (bestSolution.tse === this.tse) {
            break;
        }
    }
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/
/**
 * Get initial solutions and configure the algorithm
 */
GeneticAlgorithm.prototype.init = function () {
    this.computeCrossSetSize(this.S);
    this.generateInitialSolutions(this.X, this.S, this.M, this.solutions);
    this.sortSolutions(this.solutions);
};

/**
 * Compute cross set size
 * @param S
 */
GeneticAlgorithm.prototype.computeCrossSetSize = function (S) {
    var s = 0;

    while (s * (s + 1) / 2 < S) {
        s++;
    }

    this.crossSetSize = s;
};

/**
 * Generate initial solutions
 * @param X
 * @param S
 * @param M
 * @param solutions
 */
GeneticAlgorithm.prototype.generateInitialSolutions = function (X, S, M, solutions) {
    for (var i = 0; i < S; i++) {
        var C = this.generateRandomCodebook(X, M);
        var P = this.generateOptimalPartition(X, C);
        var tse = this.sumSquaredError(X, C, P);
        solutions[i] = {
            "codebook": C,
            "partition": P,
            "tse": tse
        };
    }
};

/**
 * Generate random code book
 * @param X
 * @param M
 * @returns {Array}
 */
GeneticAlgorithm.prototype.generateRandomCodebook = function (X, M) {
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
 * Generate optimal partition from codebook
 * @param X
 * @param C
 * @returns {Array}
 */
GeneticAlgorithm.prototype.generateOptimalPartition = function (X, C) {
    var P = [];

    for (var i = 0; i < this.N; i++) {
        var j = this.findNearestVector(X[i], C);
        P[i] = j;
        C[j].size = C[j].size + 1;
    }

    return P;
};

/**
 * Find nearest vector from a set of vectors
 * @param x
 * @param V
 * @returns {number}
 */
GeneticAlgorithm.prototype.findNearestVector = function (x, V) {
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

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/
/**
 * Generate new solutions from current solution population
 * @param X
 * @param M
 * @param S
 * @param solutions
 * @returns {Array}
 */
GeneticAlgorithm.prototype.generateNewSolutions = function (X, M, S, solutions) {
    var a = 0, b = 0;
    var newSolutions = [];

    for (var i = 0; i < S; i++) {
        var pair = this.selectNextPair(a, b);
        a = pair[0];
        b = pair[1];

        var S1 = solutions[a];
        var S2 = solutions[b];

        var Ca = S1.codebook;
        var Pa = S1.partition;

        var Cb = S2.codebook;
        var Pb = S2.partition;

        var crossedSolution = this.crossSolutions(X, M, Ca, Pa, Cb, Pb);

        newSolutions[i] = this.iterateByKMeans(X, M, crossedSolution.codebook, crossedSolution.partition);
    }

    return newSolutions;
};

/**
 * Cross the solution using PNN
 * @param X
 * @param M
 * @param C1
 * @param P1
 * @param C2
 * @param P2
 * @returns {{codebook: (Array.<T>|string), partition: Array}}
 */
GeneticAlgorithm.prototype.crossSolutions = function (X, M, C1, P1, C2, P2) {
    var CNew = this.combineCentroids(C1, C2);
    var PNew = this.combinePartitions(X, P1, P2, C1, C2);
    CNew = this.updateCentroids(X, CNew, PNew);

    this.removeEmptyClusters(X, CNew, PNew, M);
    this.performPNN(X, M, CNew, PNew);

    return {
        "codebook": CNew,
        "partition": PNew
    }
};

/**
 * Remove empty clusters
 * @param X
 * @param C
 * @param P
 * @param M
 */
GeneticAlgorithm.prototype.removeEmptyClusters = function (X, C, P, M) {
    for (var i = 0; i < C.length; i++) {
        if (C[i].size === 0 && C.length > M) {
            var last = C.length - 1;
            var tmpSize = C[i].size;
            C[i] = C[last].clone();
            C[i].size = tmpSize;
            this.joinPartitions(P, C, i, last);
            C.length--;
        }
    }
};

/**
 * Join two partitions together
 * @param X
 * @param P
 * @param C
 * @param a
 * @param b
 */
GeneticAlgorithm.prototype.joinPartitions = function (P, C, a, b) {
    for (var i = 0; i < this.N; i++) {
        if (P[i] === b) {
            P[i] = a;
        }
    }
    C[a].size = C[a].size + C[b].size;
};

/**
 * Update centroids
 * @param X
 * @param C
 * @param P
 * @returns {*}
 */
GeneticAlgorithm.prototype.updateCentroids = function (X, C, P) {
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
            C[k].size = count[k];
        } else {
            C[k].size = 0;
        }
    }

    return C;
};

/**
 * Combine two codebooks from parent solutions
 * @param C1
 * @param C2
 * @returns {Array.<T>|Buffer|string}
 */
GeneticAlgorithm.prototype.combineCentroids = function (C1, C2) {
    return C1.concat(C2);
};

/**
 * Combine two partitions from parent solutions
 * @param X
 * @param P1
 * @param P2
 * @param C1
 * @param C2
 * @returns {Array}
 */
GeneticAlgorithm.prototype.combinePartitions = function (X, P1, P2, C1, C2) {
    var m = C1.length - 1;
    var PNew = [];
    for (var i = 0; i < this.N; i++) {
        var p1 = P1[i];
        var p2 = P2[i];
        var d1 = this.distance(X[i], C1[p1], true);
        var d2 = this.distance(X[i], C2[p2], true);
        if (d1 < d2) {
            PNew[i] = p1;
        } else {
            PNew[i] = p2 + m;
        }
    }
    return PNew;
};

/**
 * Select next pair of solutions for crossing
 * @param a
 * @param b
 * @returns {[*,*]}
 */
GeneticAlgorithm.prototype.selectNextPair = function (a, b) {
    b++;

    if (b === this.crossSetSize) {
        a++;
        b = a + 1;
    }

    return [a, b];
};

/********************************************************************
 * K-MEANS ROUTINES                                                 *
 ********************************************************************/
/**
 * Iterate by K-Means to fine tune the solution
 * @param X
 * @param M
 * @param C
 * @param P
 * @returns {{codebook: *, partition: *, tse: number}}
 */
GeneticAlgorithm.prototype.iterateByKMeans = function (X, M, C, P) {
    var active = [];
    var changedList = [-1];

    var iterations = 0;

    while (iterations < this.maxKMeansIterations && changedList.length > 0) {
        var CPrev = C.clone();
        C = this.calculateCentroids(X, C, P);
        this.detectChangedCodeVectors(CPrev, C, active, changedList);
        P = this.reducedSearchPartition(X, C, P, active, changedList);
        iterations++;
    }

    return {
        "codebook": C,
        "partition": P,
        "tse": this.sumSquaredError(X, C, P)
    }
};

/**
 * Calculate K-Means centroids
 * @param X
 * @param C
 * @param P
 * @returns {Array}
 */
GeneticAlgorithm.prototype.calculateCentroids = function (X, C, P) {
    var newCodebook = [];
    var K = C.length;

    for (var i = 0; i < K; i++) {

        var indices = P.allIndexOf(i);
        var vectors = X.getElementsByIndices(indices);

        // Default to old centroid
        var centroid = C[i];

        if (vectors.length > 0) { // If the list of vectors is not empty
            centroid = this.calculateMeanVector(vectors);
        }

        newCodebook[i] = centroid;
        newCodebook[i].size = 0;
    }

    return newCodebook;
};

/**
 * Detect code vectors changes
 * @param CPrev
 * @param CNew
 * @param active
 * @param changedList
 */
GeneticAlgorithm.prototype.detectChangedCodeVectors = function (CPrev, CNew, active, changedList) {
    changedList.length = 0;
    var K = CPrev.length;

    for (var j = 0; j < K; j++) {
        active[j] = false;
        if (!CPrev[j].equals(CNew[j])) {
            if (changedList.indexOf(j) === -1) {
                changedList.push(j);
            }
            active[j] = true;
        }
    }

};

/**
 * Reduced search partition
 * @param X
 * @param C
 * @param P
 * @param active
 * @param changedList
 * @returns {*}
 */
GeneticAlgorithm.prototype.reducedSearchPartition = function (X, C, P, active, changedList) {
    var k = -1;

    for (var i = 0; i < this.N; i++) {

        if (changedList.length > 1) {

            var j = P[i];
            if (active[j]) {
                k = this.findNearestVector(X[i], C);
            } else {
                k = this.findNearestInSet(X[i], C, changedList);
            }

        } else {

            k = this.findNearestVector(X[i], C);

        }

        P[i] = k;
        C[k].size += 1;
    }

    return P;
};

/**
 * Find nearest vectors in the changed list
 * @param vector
 * @param C
 * @param changedList
 * @returns {number}
 */
GeneticAlgorithm.prototype.findNearestInSet = function (vector, C, changedList) {
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
GeneticAlgorithm.prototype.performPNN = function (X, M, C, P) {
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
GeneticAlgorithm.prototype.updatePointers = function (C, Q) {
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
GeneticAlgorithm.prototype.mergeVectors = function (X, C, P, Q, a, b) {
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
    this.fillEmptyPosition(C, Q, b, last);
    C.length--;
};

/**
 * Fill empty positions
 * @param C
 * @param Q
 * @param b
 * @param last
 */
GeneticAlgorithm.prototype.fillEmptyPosition = function (C, Q, b, last) {
    if (b !== last) {
        C[b] = C[last].clone();
        C[b].size = C[last].size;
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
GeneticAlgorithm.prototype.markClustersForRecalculation = function (C, Q, a, b) {
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
GeneticAlgorithm.prototype.createCentroid = function (C1, C2) {
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
GeneticAlgorithm.prototype.findMinimumDistance = function (C, Q) {
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
GeneticAlgorithm.prototype.findNearestNeighbor = function (C, a) {
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
GeneticAlgorithm.prototype.mergeDistortion = function (C1, C2) {
    var n1 = C1.size + 1;
    var n2 = C2.size + 1;
    var factor = (n1 * n2) / (n1 + n2);
    var distance = this.distance(C1, C2, true);
    return factor * distance;
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
GeneticAlgorithm.prototype.calculateDissimilarity = function (s1, s2) {

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
GeneticAlgorithm.prototype.centroidIndex = function (s1, s2) {

    if (!s1 || !s2 || s1.length <= 0 || s2.length <= 0) {
        throw "Invalid input!";
    }

    var CI1 = this.calculateDissimilarity(s1, s2);
    var CI2 = this.calculateDissimilarity(s2, s1);
    return Math.max(CI1, CI2);
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
GeneticAlgorithm.prototype.distance = function (x1, x2, squared) {
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
 * Calculate the mean/average vector from a set of vectors
 * @param vectors
 */
GeneticAlgorithm.prototype.calculateMeanVector = function (vectors) {
    var sumVector = vectors[0];
    var nVectors = vectors.length;
    for (var i = 1; i < vectors.length; i++) {
        sumVector = sumVector.addArray(vectors[i]);
    }
    return sumVector.divideBy(nVectors);
};
/**
 * Sort solutions based on their SSE
 * Smallest SSE value means better solution
 */
GeneticAlgorithm.prototype.sortSolutions = function (solutions) {
    solutions.sort(function (a, b) {
        return a.tse - b.tse;
    });
};
/**
 * Store the best solution
 * @param bestSolution
 */
GeneticAlgorithm.prototype.storeBestSolution = function (bestSolution) {
    this.centroids = bestSolution.codebook.clone();
    this.clusterLabels = bestSolution.partition.clone();
    this.tse = bestSolution.tse;
    this.nmse = this.normalisedMeanSquareError(this.X, this.centroids, this.clusterLabels, this.tse);
    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(this.centroids, this.GT);
    }
};
/**
 * Print codebook structure as text
 * @param C
 * @returns {string}
 */
GeneticAlgorithm.prototype.printCodebook = function (C) {
    var text = "";
    var len = C.length;
    for (var i = 0; i < len; i++) {
        text += "[" + C[i].toString() + "]{" + C[i].size + "}";
        if (i < len - 1) {
            text += ", ";
        }
    }
    return text;
};
/**
 * Print partition as text
 * @param P
 * @returns {string}
 */
GeneticAlgorithm.prototype.printPartition = function (P) {
    var text = "[";
    var len = P.length;
    for (var i = 0; i < len; i++) {
        text += P[i];
        if (i < len - 1) {
            text += ", ";
        }
    }
    text += "]";
    return text;
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
GeneticAlgorithm.prototype.sumSquaredError = function (X, C, P) {
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
GeneticAlgorithm.prototype.normalisedMeanSquareError = function (X, C, P, tse) {

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
