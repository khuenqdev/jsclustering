/**
 * Created by Nguyen Quang Khue on 18-May-17.
 * Demonstrate usage of clustering algorithms
 * Best run using Node.js
 */

const AL_MS = 0;
const AL_GA = 1;
const AL_KM = 2;

var fs = require('fs');
var MeanShift = require('./algorithms/MeanShift.js');
var GeneticAlgorithm = require('./algorithms/GeneticAlgorithm.js');
var KMeans = require('./algorithms/KMeans.js');

/**
 * Main execution point
 * @param algorithm
 * @param datafile
 * @param params
 */
function execute(algorithm, datafile, params) {

    var rawData = fs.readFileSync('data/' + datafile + '.txt', 'utf8');
    var rawGroundTruths = fs.readFileSync('data/' + datafile + '-gt.txt', 'utf8');
    var sampleData = parseData(rawData);
    var groundTruths = parseData(rawGroundTruths);

    // Default parameters
    if (!params) {
        params = {};
    }

    if (typeof params.algorithm_repeat === "undefined") {
        params.algorithm_repeat = 100;
    }
    if (typeof params.no_of_clusters === "undefined") {
        params.no_of_clusters = groundTruths.length;
    }
    if (typeof params.ms_radius === "undefined") {
        params.ms_radius = 0;
    }
    if (typeof params.km_max_iter === "undefined") {
        params.km_max_iter = 100;
    }
    if (typeof params.ga_population === "undefined") {
        params.ga_population = 45;
    }
    if (typeof params.ga_max_iter === "undefined") {
        params.ga_max_iter = 50;
    }

    var sse = 0;
    var nMSE = 0;
    var ci = 0;
    var success = 0;
    var stopIter = 0;
    var time = 0;
    var nClusters = 0;

    var bot;
    var algorithmName = "";

    var sseArr = [];
    var executionTimes = [];

    switch (algorithm) {
        case AL_MS:
            algorithmName = "Mean Shift";
            break;
        case AL_GA:
            algorithmName = "Genetic Algorithm";
            break;
        case AL_KM:
            algorithmName = "Fast K-Means";
            break;
        default:
            algorithmName = "Fast K-Means";
            break;
    }

    console.log("Running --" + algorithmName + "-- on data set [" + datafile + "]...");
    process.stdout.write("Completed 0%\r");

    var itText = algorithmName + " on [" + datafile + "]\n";
    fs.writeFileSync("logs.txt", itText, {'flag': 'a'});

    for (var i = 0; i < params.algorithm_repeat; i++) {
        var iter = i + 1;
        var hrstart = process.hrtime();

        bot = botFactory(algorithm, sampleData, groundTruths, params);
        bot.execute();

        var hrend = process.hrtime(hrstart);
        var exTimeText = hrend[0] + "s";
        executionTimes.push(hrend[0]);
        time += hrend[0];

        sse += bot.tse;
        nMSE += bot.nmse;
        ci += bot.ci;
        stopIter += bot.stopIter;
        nClusters += bot.centroids.length;
        sseArr.push(sse);

        if (bot.ci === 0) {
            success++;
        }

        process.stdout.write("Completed " + ((iter * 100) / params.algorithm_repeat) + "%\r");
        itText = "(" + iter + ") nClusters:" + bot.centroids.length + "; " +
            "SSE: " + bot.tse + "; " +
            "nMSE: " + bot.nmse + "; " +
            "CI: " + bot.ci + "; " +
            "Iteration: " + bot.stopIter + "; " +
            "Time: " + exTimeText + "\n";
        fs.writeFileSync("logs.txt", itText, {'flag': 'a'});
    }
    fs.writeFileSync("logs.txt", "-------\n", {'flag': 'a'});

    console.log("***");
    console.log("--- RESULTS FROM DATA SET [" + datafile + "] ---");
    console.log("Number of clusters: " + (nClusters / params.algorithm_repeat) + "/" + groundTruths.length);
    console.log("SSE/TSE: " + (sse / params.algorithm_repeat));
    console.log("nMSE: " + (nMSE / params.algorithm_repeat));
    console.log("CI: " + (ci / params.algorithm_repeat));
    console.log("Iterations: " + (stopIter / params.algorithm_repeat));
    console.log("Execution time: " + (time / params.algorithm_repeat));
    console.log("Success: " + ((success / params.algorithm_repeat) * 100) + "%");
    console.log("--------------------------------------------------");

    var resultTexts = "--- RESULTS FROM DATA SET [" + datafile + "] ---\n" +
        "Number of clusters: " + (nClusters / params.algorithm_repeat) + "/" + groundTruths.length + "\n" +
        "SSE/TSE: " + (sse / params.algorithm_repeat) + "\n" +
        "nMSE: " + (nMSE / params.algorithm_repeat) + "\n" +
        "CI: " + (ci / params.algorithm_repeat) + "\n" +
        "Iterations: " + (stopIter / params.algorithm_repeat) + "\n" +
        "Execution time: " + (time / params.algorithm_repeat) + "\n" +
        "Success: " + ((success / params.algorithm_repeat) * 100) + "%\n" +
        "SSEs: [" + sseArr.toString() + "]\n" +
        "Times: [" + executionTimes.toString() + "]\n" +
        "--------------------------------------------------\n";

    fs.writeFileSync("results.txt", resultTexts, {'flag': 'a'});
}

/**
 * Parse the text data to obtain real values for each feature vector
 * @param d
 */
function parseData(d) {
    var vectors = d.split("\n");
    var pdata = [];
    for (var i = 0; i < vectors.length; i++) {
        var vector = vectors[i];
        vector = vector.split(/\s+/).filter(function (feature) {
            return feature !== "" && feature !== undefined
        });
        if (vector !== undefined && vector.length > 0) {
            for (var j = 0; j < vector.length; j++) {
                vector[j] = parseFloat(vector[j]);
            }
            pdata.push(vector);
        }
    }

    return pdata;
}

/**
 * Factory to create clustering bot
 * @param algorithm
 * @param sampleData
 * @param groundTruths
 * @param params
 * @returns {*}
 */
function botFactory(algorithm, sampleData, groundTruths, params) {
    switch (algorithm) {
        case AL_MS:
            return new MeanShift.MeanShift(sampleData, params.ms_radius, params.no_of_clusters, groundTruths);
        case AL_GA:
            return new GeneticAlgorithm.GeneticAlgorithm(sampleData, params.no_of_clusters, params.ga_population, params.ga_max_iter, groundTruths);
        case AL_KM:
            return new KMeans.KMeans(sampleData, params.no_of_clusters, params.km_max_iter, groundTruths);
        default:
            return new KMeans.KMeans(sampleData, params.no_of_clusters, Infinity, groundTruths);
    }
}

/**
 * Main entry point
 */
var dataset = [
    ["dim032", {"algorithm_repeat": 100}],
    ["unbalance", {"algorithm_repeat": 100}],
    ["s1", {"algorithm_repeat": 100}],
    ["s2", {"algorithm_repeat": 100}],
    ["s3", {"algorithm_repeat": 100}],
    ["s4", {"algorithm_repeat": 100}],
    ["a1", {"algorithm_repeat": 100}],
    ["a2", {"algorithm_repeat": 100}],
    ["a3", {"algorithm_repeat": 100}],
    ["birch1", {"algorithm_repeat": 100}],
    ["birch2", {"algorithm_repeat": 100}]
];

fs.truncateSync("logs.txt");
fs.truncateSync("results.txt");

for (var j = 0; j < dataset.length; j++) {
    execute(AL_GA, dataset[j][0], dataset[j][1]);
}
