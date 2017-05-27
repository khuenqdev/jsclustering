/**
 * Created by Nguyen Quang Khue on 06-May-17.
 */

var sampledata = [];
var centroids = [];
var groundTruths = [];
var clusterLabels = [];
var tse = Infinity;
var nmse = Infinity;
var ci = Infinity;

/**
 * Some constants
 */
var K_MEANS = 0;
var MEAN_SHIFT = 1;
var GEN_AL = 2;
var PNN_AL = 3;

/********************************************************************
 * MAIN ROUTINES                                                    *
 ********************************************************************/

/**
 * Perform mean shift clustering algorithm
 * on the set of feature vectors
 */
function execute() {
    // Query I/O elements
    var statusElement = jQuery("#status");
    var nClustersElement = jQuery("#no_of_clusters");
    var algorithmElement = jQuery("#algorithm");
    var algorithm = parseInt(algorithmElement.val());
    var nClusters = parseInt(nClustersElement.val());
    var radEl = jQuery("#radius");
    var popEl = jQuery("#population_size");
    var nIterEl = jQuery("#n_iterations");
    var size = parseInt(popEl.val());
    var iter = parseInt(nIterEl.val());
    var kernelRadius = parseFloat(radEl.val());

    if (sampledata.length === 0) {
        statusElement.text("Please load one data set!");
        alert("Please load one data set!");
    } else if (algorithm !== MEAN_SHIFT && (nClusters <= 0 || !nClusters || isNaN(nClusters))) {
        statusElement.text("Please specified number of clusters!");
        alert("Please specified number of clusters!");
    } else if (algorithm === GEN_AL && (size <= 0 || iter <= 0)) {
        statusElement.text("Either population size and number of iterations is invalid!");
        alert("Either population size and number of iterations is invalid!");
    } else {
        statusElement.html("<span style='color:red'>Running " + jQuery("#algorithm[selected='selected']").text() + " algorithm...</span>");
        setTimeout(function () {
            var bot;
            switch (algorithm) {
                case K_MEANS: // Fast K-Means
                    bot = new KMeans(sampledata, nClusters, 200, groundTruths);
                    break;
                case MEAN_SHIFT: // Mean Shift
                    bot = new MeanShift(sampledata, kernelRadius, nClusters, groundTruths);
                    break;
                case GEN_AL: // Genetic Algorithm
                    bot = new GeneticAlgorithm(sampledata, nClusters, size, iter, groundTruths);
                    break;
                case PNN_AL: // Fast Pair-wise nearest neighbor
                    bot = new PNN(sampledata, nClusters, groundTruths);
                    break;
                default:
                    bot = new KMeans(sampledata, nClusters, 200, groundTruths);
                    break;
            }

            bot.execute();

            centroids = bot.centroids;
            clusterLabels = bot.clusterLabels;
            tse = bot.tse;
            nmse = bot.nmse;
            ci = bot.ci;
            stopIter = bot.stopIter;

            // Plot the centroids
            var newData = sampledata.slice(0, sampledata.length);
            initPlot(newData);
            gScatterPlot(newData, clusterLabels, {cx: false, cy: false, name: "Data Points", elClass: "datapoints"});
            newData = newData.concat(centroids);
            scatterPlot(newData, {cx: false, cy: false, r: 4, fill: 'red', name: "Centroids", elClass: "centroids"});

            if (legends.length === 0) {
                svg.append("g")
                    .attr("class", "legend")
                    .attr("transform", "translate(725,370)")
                    .attr("data-style-padding", 10)
                    .call(d3.legend);
            } else {
                svg.selectAll("g[class='legend'").remove();
                svg.append("g")
                    .attr("class", "legend")
                    .attr("transform", "translate(725,370)")
                    .attr("data-style-padding", 10)
                    .call(d3.legend);
            }

            var tseText = tse.toExponential(2).toString();
            var nMseText = tse.toExponential(2).toString();
            var p1 = tseText.substr(tseText.lastIndexOf("+") + 1);
            var p2 = nMseText.substr(tseText.lastIndexOf("+") + 1);
            tseText = tseText.replace("e+" + p1, " x 10<sup>" + p1 + "</sup>");
            nMseText = nMseText.replace("e+" + p2, "x 10<sup>" + p2 + "</sup>");

            // Print evaluation scores to the screen
            jQuery('#results_panel').append("<div id='eva_scores' class='clustering_results'>" +
                "<b>SSE/TSE:</b> " + tseText + "<br/>" +
                "<b>nMSE: </b>" + nMseText + "<br/>" +
                "<b>CI: </b>" + ci + "<br/>" +
                "<b>Iterations: </b>" + bot.stopIter + "" +
                "</div>");

            if (bot.R) {
                jQuery("#radius").val(Math.sqrt(bot.R));
            }

            statusElement.html("<span style='color:green'>Done clustering!</span>");
        }, 300);

        var legends = svg.selectAll("g[class='legend'");


    }
}

/********************************************************************
 * LOAD & PROCESS DATA                                              *
 ********************************************************************/

/**
 * Load data to memory from local predefined datasets
 * @param path local path to data set
 * @param groundTruthPath local path to ground truths data
 */
function loadDataSet(path, groundTruthPath) {
    var limitFactor = parseFloat(jQuery("#limit_factor").val()) / 100;

    if (limitFactor > 1 || limitFactor < 0) {
        alert("Invalid sub-sampling parameter, please enter number between 0 and 100!")
    } else {
        jQuery.ajax({
            method: "GET",
            url: path,
            datatype: "text"
        }).done(function (d) {
            sampledata = parseData(d, limitFactor);
            jQuery("#status").text("Data loaded!");
            alert("Data loaded!");

            // Plot the loaded data
            initPlot(sampledata);
            scatterPlot(sampledata, {cx: false, cy: false, fill: "black", name: "Data Points"});
        });

        if (groundTruthPath) {
            jQuery.ajax({
                method: "GET",
                url: groundTruthPath,
                datatype: "text"
            }).done(function (d) {
                groundTruths = parseData(d, limitFactor);
                jQuery("#status").append(" Ground truth centroids loaded!");
            });
        }
    }

}

/**
 * Parse the text data to obtain real values for each feature vector
 * @param d
 * @param limitFactor
 */
function parseData(d, limitFactor) {
    var vectors = d.split("\n");
    var pdata = [];

    jQuery.each(vectors, function (i, v) {
        v = v.split(/\s+/).filter(function (feature) {
            return feature !== "" && feature !== undefined
        });
        if (v !== undefined && v.length > 0) {
            jQuery.each(v, function (index, feature) {
                v[index] = parseFloat(feature);
            });
            pdata.push(v);
        }
    });

    if (limitFactor) {
        return pdata.slice(0, Math.round(pdata.length * limitFactor));
    } else {
        return pdata;
    }
}

/**
 * A demo with small data set
 */
function demo() {

    // Demo data
    sampledata = [
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
    ];

    // Plot the loaded data
    initPlot(sampledata);
    scatterPlot(sampledata, {cx: false, cy: false, fill: "black", name: "Data Points"});
}

/********************************************************************
 * FUNCTIONS USED FOR VISUALIZING DATA                              *
 ********************************************************************/

/**
 * Initialize visual plot's axes and scales with provided data
 * (Require d3js to work)
 * @param sampledata
 */
function initPlot(sampledata) {
    jQuery("#eva_scores").remove();

    // Remove old plots
    svg.selectAll("*").remove();

    // Initialize plot's scaling factor
    xScale = d3.scale.linear()
        .domain([0, d3.max(sampledata, function (d) {
            return d[0];
        })])
        .range([padding, w - padding * 2])
        .nice();

    yScale = d3.scale.linear()
        .domain([0, d3.max(sampledata, function (d) {
            return d[1];
        })])
        .range([h - padding, padding * 2])
        .nice();

    // Format the axes' values as floating point numbers
    var format = d3.format(".2s");

    // Create x and y axes
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")
        .ticks(5)
        .tickFormat(format);

    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left")
        .ticks(5)
        .tickFormat(format);

    // Append the axes to the plot
    svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (h - padding) + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + padding + ", 0)")
        .call(yAxis);
}

/**
 * Scatter a set of data points to a plot
 * @param sampledata
 * @param props properties of the plot
 */
function scatterPlot(sampledata, props) {

    if (!props) {
        props = {
            cx: false,
            cy: false,
            r: false,
            fill: "black",
            name: "Data Points",
            elClass: "datapoints"
        };
    }

    var circles = svg.selectAll("circle")
        .data(sampledata);

    circles.enter()
        .append("svg:circle")
        .attr("class", (!props.elClass) ? "datapoints" : props.elClass)
        .attr("data-legend", (!props.name) ? "Data Points" : props.name)
        .attr("data-legend-color", (!props.fill) ? "black" : props.fill)
        .attr("cx", function (d) {
            return (!props.cx) ? xScale(d[0]) : xScale(props.cx);
        })
        .attr("cy", function (d) {
            return (!props.cy) ? yScale(d[1]) : yScale(props.cy);
        })
        .attr("r", function (d) {
            var wh = w + h;
            var nPoints = sampledata.length;
            var def;
            if (wh > nPoints) {
                def = 3;
            } else {
                def = 0.8;
            }
            return (!props.r) ? def : props.r;
        })
        .attr("fill", (!props.fill) ? "black" : props.fill);
}

/**
 * Plot data points in group
 * @param sampledata
 * @param groups the group mapping of each data point
 * @param props properties of the plot
 */
function gScatterPlot(sampledata, groups, props) {

    if (!props) {
        props = {
            cx: false,
            cy: false,
            r: false,
            fill: "black",
            name: "Data Points",
            elClass: "datapoints"
        };
    }

    var nGroups = Math.max.apply(Math, groups) + 1;

    var groupColors = [
        "maroon",
        "olive",
        "green",
        "teal",
        "blue",
        "navy",
        "fuchsia",
        "purple",
        "blueviolet",
        "cadeblue",
        "chatreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "crimson",
        "darkgoldenrod",
        "darkgreen",
        "darkmagenta",
        "darkkhaki",
        "darkolivegreen",
        "darkred",
        "darkviolet",
        "dimgray",
        "hotpink",
        "orangered",
        "yellowgreen",
        "thistle",
        "springgreen",
        "tomato",
        "slategrey",
        "sienna"
    ];

    if (groupColors.length < nGroups) {
        for (var i = 0; i < nGroups; i++) {
            var r = Math.floor(Math.random() * 200);
            var g = Math.floor(Math.random() * 200);
            var b = Math.floor(Math.random() * 200);
            var color = d3.rgb(r, g, b);
            groupColors.push(color);
        }
    }

    var circles = svg.selectAll("circle")
        .data(sampledata);

    circles.enter()
        .append("svg:circle")
        .attr("class", (!props.elClass) ? "datapoints" : props.elClass)
        .attr("data-legend", (!props.name) ? "Data Points" : props.name)
        .attr("data-legend-color", (!props.fill) ? "black" : props.fill)
        .attr("cx", function (d) {
            return (!props.cx) ? xScale(d[0]) : xScale(props.cx);
        })
        .attr("cy", function (d) {
            return (!props.cy) ? yScale(d[1]) : yScale(props.cy);
        })
        .attr("r", function (d) {
            var wh = w + h;
            var nPoints = sampledata.length;
            var def;
            if (wh > nPoints) {
                def = 3;
            } else {
                def = 0.8;
            }
            return (!props.r) ? def : props.r;
        })
        .attr("fill", function (d, i) {
            var colorNumber = groups[i];
            return groupColors[colorNumber];
        });
}