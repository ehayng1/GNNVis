import * as ort from "onnxruntime-web";
import Papa from "papaparse";
import _ from "lodash";
import { imagenetClasses } from "../data/imagenet";
import { useState, useEffect } from "react";
import * as fs from "fs";
import * as csv from "csv-parser";

// import useFetch from "./useFetch";

export async function runSqueezenetModel(
  nodes: any,
  edges: any
): Promise<[any, number]> {
  const Tensor = ort.Tensor;

  // Create session and set options. See the docs here for more options:
  console.log("String the inference");
  const session = await ort.InferenceSession.create(
    "./_next/static/chunks/pages/g_model_full.onnx"
  );

  console.log("GNN session created!");

  // const tensorA = new ort.Tensor('float32', nodes, [3, 4]);
  // const tensorB = new ort.Tensor('float32', edges, [4, 3]);

  console.log("Nodes", nodes);
  console.log("Edges", edges);

  const newNode = prepareNodes(nodes); // Float32Array(12)
  const edge = prepareEdges(edges); // Float32Array(12)

  // prepare feeds. use model input names as keys.
  const feeds = {
    // nodes: new Tensor("float32", newNode, [898704, 3]),
    // edge_index: new Tensor("int64", edge, [1000, 2]),
    nodes: new Tensor("float32", newNode, [newNode.length / 3, 3]),
    edge_index: new Tensor("int64", edge, [edge.length / 2, 2]),
  };

  console.log("Running the model!", feeds);
  const results_02 = await session.run(feeds);
  console.log(results_02);
  // const outputData = await session.run(feeds);
  const output = results_02[session.outputNames[0]];
  console.log("Output: ", output);
  // console.log("Running the model2!", feeds);

  // Run inference and get results.
  var results = [0];
  var inferenceTime = 1;
  // = await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

function prepareNodes(nodes: number[][]) {
  // const float32Data = new Float32Array(99856 * 3);
  // for (let i = 0; i < 99856 * 3; i++) {
  //   float32Data[i] = 0; // convert to float
  // }
  // return float32Data;

  // const float32Data = new Float32Array(99856 * 3);
  // for (let i = 0; i < 99856; i++) {
  //   for (let j = 0; j < 3; j++) {
  //     float32Data[i] = nodes[i][j]; // convert to float
  //   }
  // }
  // Flatten the 2D array
  const flattenedArray = nodes.flat();

  // Convert the flattened array to Float32Array
  const floatArray = new Float32Array(flattenedArray);
  console.log("New node", floatArray);
  return floatArray;
}

function prepareEdges(edges: [][]) {
  const buffer = new ArrayBuffer(24);
  const bigint64 = new BigInt64Array();

  // const bigint64Data = new BigInt64Array(1000 * 2);
  // for (let i = 0; i < 1000; i++) {
  //   for (let j = 0; i < 2; j++) {
  //     // bigint64Data[i] = 58832n;
  //     bigint64Data[i] = edges[i][j];
  //   }
  // }
  // Flatten the 2D array
  // Flatten the 2D array and convert to BigInts
  const flattenedArray = edges.flat().map((num) => BigInt(num));

  // Convert the flattened array to BigInt64Array
  const bigIntArray = new BigInt64Array(flattenedArray);
  console.log("New Edge: ", bigIntArray);
  return bigIntArray;
  // return bigint64;
}

async function runInference(
  session: ort.InferenceSession,
  preprocessedData: any
): Promise<[any, number]> {
  // Get start time to calculate inference time.
  const start = new Date();
  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  console.log(feeds);

  // Run the session inference.
  const outputData = await session.run(feeds);
  // Get the end time to calculate inference time.
  const end = new Date();
  // Convert to seconds.
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];
  //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  var outputSoftmax = softmax(Array.prototype.slice.call(output.data));

  //Get the top 5 results.
  var results = imagenetClassesTopK(outputSoftmax, 5);
  console.log("results: ", results);
  return [results, inferenceTime];
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray: number[]): any {
  // Get the largest value in the array.
  const largestNumber = Math.max(...resultArray);
  // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
  const sumOfExp = resultArray
    .map((resultItem) => Math.exp(resultItem - largestNumber))
    .reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
  //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
  return resultArray.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp;
  });
}
/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities: any, k = 5) {
  const probs = _.isTypedArray(classProbabilities)
    ? Array.prototype.slice.call(classProbabilities)
    : classProbabilities;

  const sorted = _.reverse(
    _.sortBy(
      probs.map((prob: any, index: number) => [prob, index]),
      (probIndex: Array<number>) => probIndex[0]
    )
  );

  const topK = _.take(sorted, k).map((probIndex: Array<number>) => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, " "),
      probability: probIndex[0],
    };
  });
  return topK;
}
