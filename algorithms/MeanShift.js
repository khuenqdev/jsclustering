/********************************************************************
 * Created by Nguyen Quang Khue on 18-May-17.
 *
 * This file gives implementation of Mean Shift algorithm for clustering,
 * originally presented in:
 *
 * " Y. Cheng, "Mean shift, mode seeking, and clustering", IEEE Trans.
 * on Pattern analysis and Machine Intelligence, 17 (8), 790-799, 1995."
 *
 * Updated 19-May-2017
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
MeanShift.prototype.R = 0;

/**
 * Desired number of code vectors / clusters
 * @type {number}
 */
MeanShift.prototype.M = 0;

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
 * Constructor of the algorithm class
 * @param X Input data set of vectors
 * @param R Kernel radius
 * @param M Desired number of code vectors / clusters
 * @param GT Ground truth centroids data (if applicable)
 * @constructor
 */
function MeanShift(X, R, M, GT) {

    if (!X) {
        throw "Invalid input data!";
    }

    this.X = X;

    // Calculate data size beforehand
    this.N = X.length;

    this.R = R;

    if (this.M !== Infinity && this.M > 0) {
        this.M = M;
    }

    if (GT) {
        this.GT = GT;
    }
}

/**
 * Main execution entry point
 */
MeanShift.prototype.execute = function () {

    this.initializeConfigurations();

    // Repeatedly determining optimal codebook
    var codebook = this.determineOptimalCodebook();

    // Get optimal partition from the result codebook
    var partition = this.getOptimalPartition(codebook);

    // Fine tune the solution to obtain best clustering results
    var finalSolution = this.tuningSolution(codebook, partition);

    this.storeFinalSolution(finalSolution.codebook, finalSolution.partition, finalSolution.sse);
};

/********************************************************************
 * INITIALIZATION                                                   *
 ********************************************************************/

/**
 * Set default values to algorithm configurations
 * @deprecated
 */
MeanShift.prototype.initializeConfigurations = function () {
    // For future improvements
};

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/

/**
 * Determine optimal codebook
 * @returns {*}
 */
MeanShift.prototype.determineOptimalCodebook = function () {

    var radius;

    // Determine a kernel radius if it is not set
    if (!this.R) {
        radius = this.determineKernelRadius();
        this.R = radius;
    } else {
        // Assign kernel radius as fixed radius from input
        radius = this.R;
    }

    // Add all vectors from the data to the initial codebook
    var codebook = this.X.slice(0, this.X.length);

    var optimized = false;

    var iterations = 0;

    while (!optimized) {

        var prevCodebook = codebook.slice(0, codebook.length);

        codebook = this.updateCentroids(prevCodebook, radius);

        if (codebook.equals(prevCodebook)) {
            optimized = true;
        }

        this.stopIter = iterations + 1;
        iterations++;
    }

    return codebook;
};

/**
 * Get a list of vectors that is within a specified kernel
 * given a centroid vector and a list of distances between the
 * centroid and vectors in the data set
 * @param centroid
 * @param distances
 * @param radius
 * @returns {Array}
 */
MeanShift.prototype.getVectorsWithinKernelRadius = function (centroid, distances, radius) {

    var withinRadius = [];

    withinRadius.push(centroid);

    // Add vectors that are within the kernel radius
    for (var j = 0; j < this.N; j++) {

        if (distances[j] <= radius) {
            withinRadius.push(this.X[j]);
        }

    }

    return withinRadius;

};

/**
 * Determine kernel radius by
 * 1. Generate pseudo clusters from k = sqrt(data_size) vectors
 * 2. Calculate average distance between each pseudo centroid
 *    and points within its cluster
 * 3. Get average of average of all distances
 * @returns {number}
 */
MeanShift.prototype.determineKernelRadius = function () {

    // Generate random solution for K-Means
    var k = Math.floor(Math.sqrt(this.N));
    var cb = this.generateRandomCodebook(this.X, k);
    var pt = this.getOptimalPartition(cb);

    var allDist = [];

    // For each of the k pseudo clusters
    for (var i = 0; i < k; i++) {
        var indices = pt.allIndexOf(i);
        var vectors = this.X.getElementsByIndices(indices);

        // Calculate distance between vectors inside the cluster and the centroid
        if (vectors.length > 0) {
            var dist = [];
            for (var j = 0; j < vectors.length; j++) {
                dist.push(this.distance(cb[i], vectors[j], true));
            }

            // Collect average distance value from clusters that are not empty
            allDist.push(dist.sum() / dist.length);
        }
    }

    // Final kernel radius is the average of all average min distances
    return allDist.sum() / allDist.length;
};

/**
 * Generate random codebook from a set of data
 * @param data
 * @param k
 * @returns {Array}
 */
MeanShift.prototype.generateRandomCodebook = function (data, k) {
    var indices = Math.randIntList(0, data.length - 1, k, undefined, true);
    return data.getElementsByIndices(indices);
};

/**
 * Get optimal partition mapping for a particular codebook
 * @param codebook
 * @returns {Array}
 */
MeanShift.prototype.getOptimalPartition = function (codebook) {
    var partition = [];
    for (var i = 0; i < this.N; i++) {
        partition[i] = this.findNearestVector(this.X[i], codebook);
    }
    return partition;
};

/**
 * Update codebook centroids
 * @param codebook
 * @param radius
 * @returns {Array.<*>}
 */
MeanShift.prototype.updateCentroids = function (codebook, radius) {

    var newCodebook = [];

    for (var j = 0; j < codebook.length; j++) {

        var centroid = codebook[j];

        // Get distances between the centroid to all vectors in the data set
        var distances = this.getDistancesToDataVectors(centroid, this.X);

        // Gather all vectors within the kernel radius
        var withinRadius = this.getVectorsWithinKernelRadius(centroid, distances, radius);

        // Calculate mean vector and use as new centroid
        if (withinRadius.length > 0) {
            var newCentroid = this.calculateMeanVector(withinRadius);

            if (!newCodebook.hasElement(newCentroid)) {
                newCodebook.push(newCentroid);
            }
        }
    }

    // Sort the new codebook
    return newCodebook.sort();
};

/**
 * Get a list of distances from a vector to
 * vectors in a set of data vectors
 * @param data
 * @param x
 * @returns {Array}
 */
MeanShift.prototype.getDistancesToDataVectors = function (x, data) {

    var distances = [];

    for (var i = 0; i < data.length; i++) {
        if (!x.equals(data[i])) {
            distances[i] = this.distance(x, data[i], true);
        } else {
            // Prevent selecting itself as nearest neighbor if the distances is sorted for such purpose
            distances[i] = Infinity;
        }
    }

    return distances;
};

/**
 * Store the final solution
 * @param codebook
 * @param partition
 * @param sse
 */
MeanShift.prototype.storeFinalSolution = function(codebook, partition, sse) {
    this.centroids = codebook;
    this.clusterLabels = partition;
    this.tse = sse;
    this.nmse = this.normalisedMeanSquareError(codebook, partition, this.tse);

    if (this.GT && this.GT.length > 0) {
        this.ci = this.centroidIndex(codebook, this.GT);
    }
};

/********************************************************************
 * TUNING SOLUTION ROUTINES                                         *
 ********************************************************************/

/**
 * Fine tune the solution to obtain best clustering results
 * @param codebook
 * @param partition
 * @returns {{codebook: *, partition: *, sse: number}}
 */
MeanShift.prototype.tuningSolution = function (codebook, partition) {

    // Enforce number of desired clusters if needed
    if (this.M > 0 && this.M !== Infinity) {
        if (codebook.length < this.M) {
            // Generate new centroids and getting optimal partition
            codebook = this.generateCentroids(codebook, partition, this.M);
        } else if (codebook.length > this.M) {
            // Perform PNN to merge vectors
            codebook = this.performPNN(codebook, partition);
        }
    } else {
        // Remove low density clusters
        codebook = this.removeLowDensityClusters(codebook, partition);
    }

    // Re-partition
    partition = this.getOptimalPartition(codebook);

    // Iterate by K-Means to fine-tune the solution
    return this.iterateByKMeans(codebook, partition, 2);
};

/**
 * Remove low density clusters
 * @param codebook
 * @param partition
 * @returns Array
 */
MeanShift.prototype.removeLowDensityClusters = function (codebook, partition) {

    var sizes = [];

    // Get density value by calculate average number of vectors
    // in each cluster
    for (var j = 0; j < codebook.length; j++) {
        sizes[j] = partition.countVal(j);
    }

    var density = sizes.sum() / sizes.length;

    // Keep only centroids that have high cluster density
    var newCodebook = [];
    for (var k = 0; k < codebook.length; k++) {
        if (sizes[k] > density) {
            newCodebook.push(codebook[k]);
        }
    }

    return newCodebook;
};

/**
 * Generate new centroids from the current codebook
 * obtained from clustering results
 * @param codebook
 * @param partition
 * @param nClusters
 * @returns {*}
 */
MeanShift.prototype.generateCentroids = function (codebook, partition, nClusters) {

    while (codebook.length < nClusters) {
        // Select 2 random vectors in the current cluster
        for (var i = 0; i < codebook.length; i++) {

            // Break the loop and return the codebook as soon as the number
            // of centroids reach the desired number
            if (codebook.length === nClusters) {
                return codebook;
            }

            // Get all vectors within the current cluster
            var inClusterVectors = this.X.getElementsByIndices(partition.allIndexOf(i));
            var vectors, centroid;

            // If the cluster has vectors assigned to it
            if (inClusterVectors.length > 0) {
                // Select 2 random vectors
                var indices = Math.randIntList(0, inClusterVectors.length - 1, 2, undefined, true);
                vectors = inClusterVectors.getElementsByIndices(indices);

                // Get the mean of the 2 vectors as new centroid
                centroid = this.calculateMeanVector(vectors);
            } else {
                // Otherwise, choose a random vectors from the data set
                var idx = Math.randInt(0, this.N - 1);

                // Copy the vector as new centroid
                centroid = this.X[idx].slice(0, this.X[idx].length);
            }

            if (!codebook.hasElement(centroid)) {
                codebook.push(centroid);
            }
        }
    }

    return codebook;
};

/********************************************************************
 * PNN ROUTINES                                                     *
 ********************************************************************/

/**
 * Perform pair-wise nearest neighbor algorithm
 * @param codebook
 * @param partition
 */
MeanShift.prototype.performPNN = function (codebook, partition) {
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

        // Update nearest neighbor pointers
        q = this.updatePointers(mergedResults.codebook, mergedResults.partition, mergedResults.pointers);

    }

    return codebook;
};

/**
 * Find the nearest centroid in a codebook from current centroid
 * @param codebook
 * @param partition
 * @param a
 * @returns {{nearest: number, distance: Number, recalculate: boolean}}
 */
MeanShift.prototype.findNearestNeighbor = function (codebook, partition, a) {

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
MeanShift.prototype.findMinimumDistance = function (codebook, q) {

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
MeanShift.prototype.mergeVectors = function (codebook, partition, q, a, b) {

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
MeanShift.prototype.updatePointers = function (codebook, partition, q) {

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
MeanShift.prototype.markClustersForRecalculation = function (codebook, q, a, b) {

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
MeanShift.prototype.joinPartitions = function (partition, a, b) {

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
MeanShift.prototype.fillEmptyPosition = function (codebook, q, b, last) {

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
MeanShift.prototype.mergeDistortion = function (c1, c2, n1, n2) {
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
MeanShift.prototype.createCentroid = function (c1, c2, n1, n2) {
    c1 = c1.multiplyBy(n1);
    c2 = c2.multiplyBy(n2);
    return c1.addArray(c2).divideBy(n1 + n2);
};

/********************************************************************
 * K-MEANS ROUTINES                                                 *
 ********************************************************************/

/**
 * K-Means clustering
 * @param codebook
 * @param partition
 * @param maxIterations
 * @returns {{codebook: *, partition: *, sse: number}}
 */
MeanShift.prototype.iterateByKMeans = function (codebook, partition, maxIterations) {

    var active = []; // List for determining active code vectors
    var changedList = [-1]; // List for tracking code vector changes with a dummy index

    var iterations = 0;

    while (iterations < maxIterations && changedList.length > 0) {

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
MeanShift.prototype.reducedSearchPartition = function (codebook, partition, active, changedList) {

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
MeanShift.prototype.findNearestCentroidInChangedList = function (vector, codebook, changedList) {

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
 * Detect active code vector (centroids) in the code book
 * and track changes
 * @param prevCodebook
 * @param newCodebook
 * @param active
 * @param changedList
 */
MeanShift.prototype.detectChangedCodeVectors = function (prevCodebook, newCodebook, active, changedList) {

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
MeanShift.prototype.calculateCentroids = function (codebook, partition) {
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
 * LOW-LEVEL ROUTINES                                               *
 ********************************************************************/

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
 * Calculate the mean/average vector from a set of vectors
 * @param vectors
 */
MeanShift.prototype.calculateMeanVector = function (vectors) {
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
MeanShift.prototype.distance = function (x1, x2, squared) {
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
MeanShift.prototype.sumSquaredError = function (codebook, partition) {
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
MeanShift.prototype.normalisedMeanSquareError = function (codebook, partition, tse) {

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