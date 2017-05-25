/**
 * Created by Nguyen Quang Khue on 18-May-17.
 * Demonstrate usage of clustering algorithms
 * Best run using Node.js
 */

var fs = require('fs');
var MeanShift = require('./algorithms/MeanShift.js');
var GeneticAlgorithm = require('./algorithms/GeneticAlgorithm.js');
var KMeans = require('./algorithms/KMeans.js');

var rawData = fs.readFileSync('data/s1.txt', 'utf8');
var rawGroundTruths = fs.readFileSync('data/s1-gt.txt', 'utf8');
var sampleData = parseData(rawData);
var groundTruths = parseData(rawGroundTruths);

/*sampleData = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
];
groundTruths = [
    [1.1666666666666667, 1.4666666666666666],
    [7.333333333333333, 9],
    [9, 2.3333333333333335]
];*/

var bot = new GeneticAlgorithm.GeneticAlgorithm(sampleData, 15, 45, 15, groundTruths);
bot.execute();

console.log(bot.centroids);
console.log("CI: " + bot.ci);
console.log("SSE: " + bot.tse);
console.log("nMSE: " + bot.nmse);
console.log("Iterations: " + bot.stopIter);

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