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
 * Updated 18-May-2017
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
GeneticAlgorithm.prototype.M = 0;

/**
 * Solution population size
 * @type {number}
 */
GeneticAlgorithm.prototype.S = 2;

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
 * Solution population
 * @type {Array}
 */
GeneticAlgorithm.prototype.solutions = [];

/**
 * Size of the crossover set
 * @type {number}
 */
GeneticAlgorithm.prototype.crossSetSize = 0;

/**
 * Maximum number of K-Means iterations
 * used for fine tuning the solution
 * @type {number}
 */
GeneticAlgorithm.prototype.maxKMeansIterations = 2;

/**
 * Best solution's codebook
 * @type {Array}
 */
GeneticAlgorithm.prototype.centroids = [];

/**
 * Best solution's partition mappings
 * @type {Array}
 */
GeneticAlgorithm.prototype.clusterLabels = [];

/**
 * Best solution SSE score
 * @type {Number}
 */
GeneticAlgorithm.prototype.tse = Infinity;

/**
 * Normalised Mean Square Error
 * @type {Number}
 */
GeneticAlgorithm.prototype.nmse = Infinity;

/**
 * Centroid Index score
 * @type {Number}
 */
GeneticAlgorithm.prototype.ci = Infinity;

/**
 * The iteration where the algorithm stops
 * when no improvements achieved
 * @type {number}
 */
GeneticAlgorithm.prototype.stopIter = 0;

/**
 * Algorithm class constructor
 * @param X Input data set
 * @param M Desired number of code vectors / clusters
 * @param S Solution population size
 * @param T Number of iterations (maximum number of solution generations)
 * @param GT Ground truth centroids (if applicable)
 * @constructor
 */
function GeneticAlgorithm(X, M, S, T, GT) {

    if (!X || !M) {
        throw "Either the data set or the number of clusters is missing from parameter list!";
    }

    this.X = X;
    this.M = M;

    // Calculate data size beforehand
    this.N = this.X.length;

    if (S) {
        this.S = S;
    }

    if (T) {
        this.T = T;
    }

    if (GT) {
        this.GT = GT;
    }
}

/**
 * Main execution entry point
 */
GeneticAlgorithm.prototype.execute = function () {
    this.computeCrossSetSize();
    this.generateInitialSolutions();
    this.sortSolutions();

    for (var t = 0; t < this.T; t++) {
        this.generateNewSolutions();
        this.sortSolutions();
        var bestSolution = this.solutions[0];
        this.stopIter = t + 1;
        if (bestSolution.sse < this.tse) {
            this.storeBestSolution(bestSolution);
        } else if (bestSolution.sse === this.tse) {
            // Stop at first time the solution fails to improve
            break;
        }
    }
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/

/**
 * Generate initial solutions for the algorithm
 */
GeneticAlgorithm.prototype.generateInitialSolutions = function () {
    for (var s = 0; s < this.S; s++) {
        var codebook = this.generateRandomCodebook();
        var partition = this.getOptimalPartition(codebook);
        var sse = this.sumSquaredError(codebook, partition);
        this.solutions[s] = {
            "codebook": codebook,
            "partition": partition,
            "sse": sse
        }
    }
};

/**
 * Generate a random codebook
 * @returns {Array}
 */
GeneticAlgorithm.prototype.generateRandomCodebook = function () {
    var indices = Math.randIntList(0, this.N - 1, this.M, undefined, true);
    return this.X.getElementsByIndices(indices);
};

/**
 * Get optimal partition mapping for a particular codebook
 * @param codebook
 * @returns {Array}
 */
GeneticAlgorithm.prototype.getOptimalPartition = function (codebook) {
    var partition = [];
    for (var i = 0; i < this.N; i++) {
        partition[i] = this.findNearestVector(this.X[i], codebook);
    }
    return partition;
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/

/**
 * Generate new solutions from the current solution
 * population
 */
GeneticAlgorithm.prototype.generateNewSolutions = function () {

    var a = 0, b = 0, newSolutions = [];

    for (var s = 0; s < this.S; s++) {
        // Select a pair of solutions from the population
        var pairIndices = this.selectPairForCrossover(a, b);
        a = pairIndices.a;
        b = pairIndices.b;

        // Cross the selected solutions
        var crossedSolution = this.crossSolutions(a, b);
        var newCodebook = crossedSolution.codebook;
        var newPartition = crossedSolution.partition;

        // Mutate the new solution codebook (unnecessary)
        //newCodebook = this.mutateSolution(newCodebook);

        // Fine-tune the solution using K-Means
        newSolutions[s] = this.iterateByKMeans(newCodebook, newPartition);
    }

    this.solutions = newSolutions;
};

/**
 * Cross two parent solutions to form a new solution
 * @param a
 * @param b
 * @returns {{codebook, partition}}
 */
GeneticAlgorithm.prototype.crossSolutions = function (a, b) {
    // Extract the two parent solutions
    var solutionA = this.solutions[a];
    var solutionB = this.solutions[b];

    // Get codebook and partition of first solution
    var ca = solutionA.codebook;
    var pa = solutionA.partition;

    // Get codebook and partition of second solution
    var cb = solutionB.codebook;
    var pb = solutionB.partition;

    // 1. Combine centroids of the 2 solutions
    var newCodebook = this.combineCentroids(ca, cb);

    // 2. Combine partitions of the 2 solutions
    var newPartition = this.combinePartitions(ca, pa, cb, pb);

    // 3. Generate optimal codebook and partition mapping from the combined codebook and partition
    newCodebook = this.updateCentroids(newCodebook, newPartition);

    // 4. Remove empty clusters
    var updated = this.removeEmptyClusters(newCodebook, newPartition);

    // 5. Perform PNN to refine the solution
    var refined = this.performPNN(updated.codebook, updated.partition);
    newCodebook = refined.codebook;
    newPartition = refined.partition;

    return {
        "codebook": newCodebook,
        "partition": newPartition
    }
};

/**
 * Remove empty clusters
 * @param codebook
 * @param partition
 */
GeneticAlgorithm.prototype.removeEmptyClusters = function (codebook, partition) {

    for (var j = 0; j < codebook.length; j++) {
        var size = partition.countVal(j);
        if (size === 0 && codebook.length > this.M) {
            var last = codebook.length - 1;
            codebook[j] = codebook[last].slice(0, codebook[last].length);
            partition = this.joinPartitions(partition, j, last);
            codebook.length--;
        }
    }

    return {
        "codebook": codebook,
        "partition": partition
    }
};

/**
 * Update centroids
 * @param codebook
 * @param partition
 * @returns {Array}
 */
GeneticAlgorithm.prototype.updateCentroids = function (codebook, partition) {

    var sum = [];
    var count = [];
    for (var i = 0; i < this.N; i++) {
        var j = partition[i];
        if (typeof sum[j] === "undefined") {
            sum[j] = this.X[i].slice(0, this.X[i].length).fill(0);
        }
        sum[j] = sum[j].addArray(this.X[i].slice(0, this.X[i].length));
        if (typeof count[j] === "undefined") {
            count[j] = 0;
        }
        count[j] += 1;
    }

    for (var k = 0; k < codebook.length; k++) {
        if (typeof sum[k] !== "undefined" && count[k] > 0) {
            codebook[k] = sum[k].divideBy(count[k]);
        }
    }

    return codebook;
};

/**
 * Combine centroids from parent solutions
 * @param ca
 * @param cb
 * @returns {Array.<T>|string}
 */
GeneticAlgorithm.prototype.combineCentroids = function (ca, cb) {
    var newCodebook = ca.slice(0, ca.length);
    return newCodebook.concat(cb);
};

/**
 * Combine partitions from parent solutions
 * @param ca
 * @param pa
 * @param cb
 * @param pb
 * @returns {Array}
 */
GeneticAlgorithm.prototype.combinePartitions = function (ca, pa, cb, pb) {

    var newPartition = [];

    for (var i = 0; i < this.N; i++) {

        // Get cluster label from each partition's corresponding vector
        var j1 = pa[i];
        var j2 = pb[i];

        // Obtain the centroid from the 2 solutions' codebooks
        var c1 = ca[j1];
        var c2 = cb[j2];

        // Calculate distances between current vector and the two centroids
        var d1 = this.distance(this.X[i], c1, true);
        var d2 = this.distance(this.X[i], c2, true);

        // Compare between the 2 distances to select matching cluster label
        if (d1 < d2) {
            newPartition[i] = j1;
        } else {
            newPartition[i] = j2 + (ca.length - 1); // Use index of the combined centroid codebook
        }

    }

    return newPartition;
};

/**
 * Mutate the current solution by selecting random
 * vector from input data as a codebook's centroid
 * @param codebook
 */
GeneticAlgorithm.prototype.mutateSolution = function (codebook) {
    var i = Math.randInt(0, this.X.length - 1);
    var j = Math.randInt(0, codebook.length - 1);
    codebook[j] = this.X[i].slice(0, this.X[i].length);
    return codebook;
};

/**
 * Select a pair of solutions for crossover
 * @param a
 * @param b
 * @returns {{a: *, b: *}}
 */
GeneticAlgorithm.prototype.selectPairForCrossover = function (a, b) {

    b++;

    if (b === this.crossSetSize) {
        a++;
        b = a + 1;
    }
    return {
        "a": a,
        "b": b
    }

};

/**
 * Calculate the cross solution set's size
 */
GeneticAlgorithm.prototype.computeCrossSetSize = function () {

    var s = 1;

    while ((s * (s + 1) / 2) < this.S) {
        s++;
    }

    this.crossSetSize = s;
};

/********************************************************************
 * K-MEANS ROUTINES                                                 *
 ********************************************************************/

/**
 * K-Means clustering
 * @param codebook
 * @param partition
 * @returns {{codebook: *, partition: *, sse: number}}
 */
GeneticAlgorithm.prototype.iterateByKMeans = function (codebook, partition) {

    var active = []; // List for determining active code vectors
    var changedList = [-1]; // List for tracking code vector changes with a dummy index

    var iterations = 0;

    while (iterations < this.maxKMeansIterations && changedList.length > 0) {

        var prevCodebook = codebook.slice(0, codebook.length); // Take a snapshot of last codebook

        codebook = this.calculateCentroids(prevCodebook, partition);

        // Detect changes and active centroids
        var changes = this.detectChangedCodeVectors(prevCodebook, codebook, active, changedList);
        changedList = changes.changedList;
        active = changes.activeList;

        partition = this.reducedSearchPartition(codebook, partition, active, changedList);

        iterations++;

    }

    // Output the fine-tuned solution
    var sse = this.sumSquaredError(codebook, partition);
    return {
        "codebook": codebook,
        "partition": partition,
        "sse": sse
    }
};

/**
 * Reduce the search partition by updating cluster labels
 * of each input data vector to the nearest code vector (centroid)
 * @param codebook
 * @param partition
 * @param active
 * @param changedList
 * @returns {*}
 */
GeneticAlgorithm.prototype.reducedSearchPartition = function (codebook, partition, active, changedList) {

    // For each input data vector
    for (var i = 0; i < this.N; i++) {

        if (changedList.length > 1) {
            var j = partition[i]; // Get its current cluster label in the partition mapping

            if (active[j]) { // If the code vector corresponding to the cluster is active
                // Find and assign the current vector to the cluster of the nearest code vector
                partition[i] = this.findNearestVector(this.X[i], codebook);
            } else {
                // Otherwise, find and assign the current vector to the cluster of the nearest code vector in the active code vector list
                partition[i] = this.findNearestCentroidInChangedList(this.X[i], codebook, changedList);
            }
        } else {
            partition[i] = this.findNearestVector(this.X[i], codebook);
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
GeneticAlgorithm.prototype.findNearestCentroidInChangedList = function (vector, codebook, changedList) {

    var minDist = Infinity;
    var minIndex = 0;
    var l = changedList.length;

    for (var i = 0; i < l; i++) {
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
 * Detect active code vector (centroids) in the code book
 * and track changes
 * @param prevCodebook
 * @param newCodebook
 * @param active
 * @param changedList
 */
GeneticAlgorithm.prototype.detectChangedCodeVectors = function (prevCodebook, newCodebook, active, changedList) {

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
 * Calculate partition centroids and uses them as code vectors
 * @param codebook
 * @param partition
 * @returns {Array}
 */
GeneticAlgorithm.prototype.calculateCentroids = function (codebook, partition) {
    var newCodebook = [];

    for (var i = 0; i < codebook.length; i++) {

        var indices = partition.allIndexOf(i);
        var vectors = this.X.getElementsByIndices(indices);

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
 * PNN ROUTINES                                                     *
 ********************************************************************/

/**
 * Perform pair-wise nearest neighbor algorithm
 * @param codebook
 * @param partition
 */
GeneticAlgorithm.prototype.performPNN = function (codebook, partition) {
    var q = []; // Nearest neighbor pointers

    // For each centroid in the codebook
    for (var j = 0; j < codebook.length; j++) {
        // Find its nearest neighbor
        q[j] = this.findNearestNeighbor(codebook, partition, j);
    }

    // Repeat until getting desired codebook size
    while (codebook.length > this.M) {

        // Find the index of centroid with distance to another centroid to be the smallest
        var a = this.findMinimumDistance(codebook, q);

        // Get the index of the nearest centroid of that centroid
        var b = q[a].nearest;

        // Merge the two vectors together
        var mergedResults = this.mergeVectors(codebook, partition, q, a, b);
        codebook = mergedResults.codebook;
        partition = mergedResults.partition;
        q = mergedResults.pointers;

        // Update nearest neighbor pointers
        q = this.updatePointers(codebook, partition, q);

    }

    return {
        "codebook": codebook,
        "partition": partition
    }
};

/**
 * Find the nearest centroid in a codebook from current centroid
 * @param codebook
 * @param partition
 * @param a
 * @returns {{nearest: number, distance: Number, recalculate: boolean}}
 */
GeneticAlgorithm.prototype.findNearestNeighbor = function (codebook, partition, a) {

    var q = {
        "nearest": 0,
        "distance": Infinity,
        "recalculate": false
    };

    for (var j = 0; j < codebook.length; j++) {

        // Get the size of each cluster by counting the number of vectors assigned to it
        var n1 = partition.countVal(a) + 1;
        var n2 = partition.countVal(j) + 1;

        var d = this.mergeDistortion(codebook[a], codebook[j], n1, n2);

        if (a !== j && d < q.distance) {
            q.nearest = j;
            q.distance = d;
        }

    }

    return q;

};

/**
 * Find the index of centroid that has distance to
 * its nearest centroid to be the smallest
 * @param codebook
 * @param q
 * @returns {number}
 */
GeneticAlgorithm.prototype.findMinimumDistance = function (codebook, q) {

    var minDist = Infinity;
    var minIndex = 0;

    for (var j = 0; j < codebook.length; j++) {

        if (q[j].distance < minDist) {
            minIndex = j;
            minDist = q[j].distance;
        }

    }

    return minIndex;

};

/**
 * Merge 2 centroid vectors
 * @param codebook
 * @param partition
 * @param q
 * @param a
 * @param b
 */
GeneticAlgorithm.prototype.mergeVectors = function (codebook, partition, q, a, b) {

    if (a > b) {
        // Swap a & b so that a is smaller index
        var tmp = a;
        a = b;
        b = tmp;
    }

    // Get last index of the codebook
    var last = codebook.length - 1;

    // Mark clusters for recalculation
    q = this.markClustersForRecalculation(codebook, q, a, b);

    // Create a new centroid vector as weighted average vector of centroid a & b
    var n1 = partition.countVal(a) + 1;
    var n2 = partition.countVal(b) + 1;
    codebook[a] = this.createCentroid(codebook[a], codebook[b], n1, n2);

    // Join the partitions of b to a
    partition = this.joinPartitions(partition, a, b);

    // Fill empty position with the last centroid
    var filledData = this.fillEmptyPosition(codebook, q, b, last);
    codebook = filledData.codebook;
    q = filledData.pointers;

    // Decrease codebook size
    codebook.length--;

    return {
        "codebook": codebook,
        "partition": partition,
        "pointers": q
    };

};

/**
 * Update nearest neighbor pointers
 * @param codebook
 * @param partition
 * @param q
 * @returns {*}
 */
GeneticAlgorithm.prototype.updatePointers = function (codebook, partition, q) {

    // If any of the nearest neighbor pointers needs to be recalculated
    for (var j = 0; j < codebook.length; j++) {
        if (q[j].recalculate === true) {
            q[j] = this.findNearestNeighbor(codebook, partition, j);
            q[j].recalculate = false;
        }
    }

    return q;
};

/**
 * Mark the clusters that need recalculation by
 * determining whether the cluster nearest neighbor
 * is vector a or b
 * @param codebook
 * @param q
 * @param a
 * @param b
 */
GeneticAlgorithm.prototype.markClustersForRecalculation = function (codebook, q, a, b) {

    for (var j = 0; j < codebook.length; j++) {
        q[j].recalculate = q[j].nearest === a || q[j].nearest === b;
    }

    return q;
};

/**
 * Join 2 partitions a and b so that all vectors will be in a
 * and cluster b will be empty. Pointers are updated
 * @param partition
 * @param a
 * @param b
 */
GeneticAlgorithm.prototype.joinPartitions = function (partition, a, b) {

    for (var i = 0; i < this.N; i++) {
        if (partition[i] === b) {
            partition[i] = a;
        }
    }

    return partition;
};

/**
 * Fill empty positions of the nearest neighbor mapping
 * and the codebook
 * @param codebook
 * @param q
 * @param b
 * @param last
 */
GeneticAlgorithm.prototype.fillEmptyPosition = function (codebook, q, b, last) {

    if (b !== last) {

        codebook[b] = codebook[last];
        q[b] = q[last];

        // Update pointers to point all nearest point to b
        for (var j = 0; j < codebook.length; j++) {
            if (q[j].nearest === last) {
                q[j].nearest = b;
            }
        }

    }

    return {
        "codebook": codebook,
        "pointers": q
    };

};

/**
 * Calculate the merge cost of the cluster using equation (4)
 * in the paper
 * @param c1 the first centroid vector
 * @param c2 the second centroid vector
 * @param n1
 * @param n2
 * @returns {number}
 */
GeneticAlgorithm.prototype.mergeDistortion = function (c1, c2, n1, n2) {
    return ((n1 * n2) * this.distance(c1, c2, true)) / (n1 + n2);
};

/**
 * Create a centroid that is the weighted average of the two
 * centroid vectors
 * @param c1 first centroid vector
 * @param c2 second centroid vector
 * @param n1 cluster size of first centroid
 * @param n2 cluster size of second centroid
 * @returns {Array}
 */
GeneticAlgorithm.prototype.createCentroid = function (c1, c2, n1, n2) {
    c1 = c1.multiplyBy(n1);
    c2 = c2.multiplyBy(n2);
    return c1.addArray(c2).divideBy(n1 + n2);
};

/********************************************************************
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/

/**
 * Store a copy of the best solution with min SSE
 */
GeneticAlgorithm.prototype.storeBestSolution = function (bestSolution) {
    this.centroids = bestSolution.codebook.slice(0, bestSolution.codebook.length);
    this.clusterLabels = bestSolution.partition.slice(0, bestSolution.partition.length);
    this.tse = bestSolution.sse;
    this.nmse = this.normalisedMeanSquareError(bestSolution.codebook, bestSolution.partition, this.tse);
    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(bestSolution.codebook, this.GT);
    }
};

/**
 * Get the nearest vector to an input vector
 * @param x
 * @param vectors
 * @returns {number}
 */
GeneticAlgorithm.prototype.findNearestVector = function (x, vectors) {

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
GeneticAlgorithm.prototype.sortSolutions = function () {
    this.solutions.sort(function (a, b) {
        return a.sse > b.sse;
    });
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
GeneticAlgorithm.prototype.sumSquaredError = function (codebook, partition) {
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
GeneticAlgorithm.prototype.normalisedMeanSquareError = function (codebook, partition, tse) {

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