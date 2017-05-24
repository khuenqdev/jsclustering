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

function execute(algorithm, datafile, maxIter, nClusters) {

    var rawData = fs.readFileSync('data/' + datafile + '.txt', 'utf8');
    var rawGroundTruths = fs.readFileSync('data/' + datafile + '-gt.txt', 'utf8');
    var sampleData = parseData(rawData);
    var groundTruths = parseData(rawGroundTruths);

    if (!nClusters) {
        nClusters = groundTruths.length;
    }

    if (!maxIter) {
        maxIter = 100;
    }

    var sse = 0;
    var nMSE = 0;
    var ci = 0;
    var success = 0;
    var stopIter = 0;
    var bot;
    var algorithmName = "";
    var sseArr = [];
    var executionTimes = [];

    switch (algorithm) {
        case AL_MS:
            bot = new MeanShift.MeanShift(sampleData, 0, nClusters, groundTruths);
            algorithmName = "Mean Shift";
            break;
        case AL_GA:
            bot = new GeneticAlgorithm.GeneticAlgorithm(sampleData, nClusters, 45, 50, groundTruths);
            algorithmName = "Genetic Algorithm";
            break;
        case AL_KM:
            algorithmName = "Fast K-Means";
            bot = new KMeans.KMeans(sampleData, nClusters, 100, groundTruths);
            break;
        default:
            algorithmName = "Fast K-Means";
            bot = new KMeans.KMeans(sampleData, nClusters, Infinity, groundTruths);
            break;
    }

    console.log("Running --" + algorithmName + "-- on data set [" + datafile + "]...");
    process.stdout.write("Completed 0%\r");

    var itText = algorithmName + " on [" + datafile + "]\n";
    fs.writeFileSync("logs.txt", itText, {'flag': 'a'});

    for (var i = 0; i < maxIter; i++) {
        var iter = i + 1;
        var hrstart = process.hrtime();

        bot.execute();

        var hrend = process.hrtime(hrstart);
        var secs = hrend[1] / 1000000000;
        var exTimeText = secs + "s";
        executionTimes.push(secs);

        sse += bot.tse;
        nMSE += bot.nmse;
        ci += bot.ci;
        stopIter += bot.stopIter;
        sseArr.push(sse);

        if (bot.ci === 0) {
            success++;
        }

        process.stdout.write("Completed " + ((iter * 100) / maxIter) + "%\r");
        itText = "(" + iter + ") SSE: " + bot.tse + "; " +
            "nMSE: " + bot.nmse + "; " +
            "CI: " + bot.ci + "; " +
            "Stop at iteration: " + bot.stopIter + "; " +
            "Time: " + exTimeText + "\n";
        fs.writeFileSync("logs.txt", itText, {'flag': 'a'});
    }
    fs.writeFileSync("logs.txt", "-------\n", {'flag': 'a'});

    console.log("***");
    console.log("--- RESULTS FROM DATA SET [" + datafile + "] ---");
    console.log("Number of clusters: " + nClusters);
    console.log("SSE/TSE: " + (sse / maxIter));
    console.log("nMSE: " + (nMSE / maxIter));
    console.log("CI: " + (ci / maxIter));
    console.log("Iterations: " + (stopIter / maxIter));
    console.log("Success: " + ((success / maxIter) * 100) + "%");

    var resultTexts = "--- RESULTS FROM DATA SET [" + datafile + "] ---\n" +
        "Number of clusters: " + nClusters + "\n" +
        "SSE/TSE: " + (sse / maxIter) + "\n" +
        "nMSE: " + (nMSE / maxIter) + "\n" +
        "CI: " + (ci / maxIter) + "\n" +
        "Iterations: " + (stopIter / maxIter) + "\n" +
        "Success: " + ((success / maxIter) * 100) + "%\n" +
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
 * Main entry point
 */
var dataset = [
    "dim032",
    "unbalance",
    "s1",
    "s2",
    "s3",
    "s4",
    "a1",
    "a2",
    "a3",
    "birch1",
    "birch2"
];

fs.truncateSync("logs.txt");
fs.truncateSync("results.txt");
for (var j = 0; j < dataset.length; j++) {
    execute(AL_GA, dataset[j], 2);
}
