import { useRef, useState, useEffect } from "react";
import { IMAGE_URLS } from "../data/sample-image-urls";
// import { inferenceSqueezenet } from "../utils/predict";
import csv from "csv-parser";
import styles from "../styles/Home.module.css";
import fs from "fs"; // If you're using Node.js
import { runSqueezenetModel } from "../utils/modelHelper";
import useFetch from "../hooks/useFetch";
import Papa from "papaparse";
import { setegid } from "process";

const acceptableCSVFileTypes =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel, .csv";

interface Props {
  height: number;
  width: number;
}
interface CSVData {
  [key: string]: string;
}

const ImageCanvas = (props: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  var image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState("");
  const [topResultConfidence, setConfidence] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");
  const [nodes, setNodes] = useState<any[]>([]);
  const [edges, setEdges] = useState<any[]>([]);
  const [data, setData] = useState<any[]>([]);
  // const [edges, setEdges] = useState();
  const { fetchCsvData } = useFetch();

  useEffect(() => {
    // fetchCsvData(
    //   "./_next/static/chunks/pages/in_g_edge_list1000.csv",
    //   setNodes
    // );
    // fetchCsvData("/data/in_g_edge_list1000.csv", setNodes);
    // fetchCsvData('/data/recipe-database.csv', setNodes)
  }, []);

  // Load the image from the IMAGE_URLS array
  const getImage = () => {
    var sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
    var random = Math.floor(Math.random() * (9 - 0 + 1) + 0);
    return sampleImageUrls[random];
  };

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = () => {
    // Get the image
    image = new Image();
    var sampleImage = getImage();
    image.src = sampleImage.value;

    // Clear out previous values.
    setLabel(`Inferencing...`);
    setConfidence("");
    setInferenceTime("");

    // Draw the image on the canvas
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext("2d");
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    };

    // Run the inference
    submitInference();
  };

  const handleCsvNodes = (event: React.ChangeEvent<HTMLInputElement>) => {
    const csvFile = event.target.files?.[0];

    if (csvFile) {
      console.log("Parsing starts!");
      Papa.parse(csvFile, {
        skipEmptyLines: true,
        header: false, // Don't treat the first row as headers
        complete: function (results) {
          // Handle parsing results
          // console.log("Nodes: ", results.data);
          setNodes(results.data);
        },
        error: function (error) {
          console.error("CSV parsing error:", error);
          // Handle parsing error
        },
      });
    } else {
      console.error("No file selected.");
      // Handle case where no file is selected
    }
  };

  const handleCsvEdges = (event: React.ChangeEvent<HTMLInputElement>) => {
    const csvFile = event.target.files?.[0];

    if (csvFile) {
      console.log("Parsing starts!");
      Papa.parse(csvFile, {
        header: false, // Don't treat the first row as headers
        dynamicTyping: true, // Automatically detect data types
        delimiter: ",", // Specify the delimiter if it's not a comma
        newline: "\n", // Specify the newline character if it's not a standard newline
        skipEmptyLines: true, // Skip empty lines
        complete: function (results) {
          // Handle parsing results
          // console.log("Edges: ", results.data);
          setEdges(results.data);
        },
        error: function (error) {
          console.error("CSV parsing error:", error);
          // Handle parsing error
        },
      });
    } else {
      console.error("No file selected.");
      // Handle case where no file is selected
    }
  };

  const submitInference = async () => {
    // Get the image data from the canvas and submit inference.
    // var [inferenceResult, inferenceTime] = await inferenceSqueezenet(image.src);
    var [inferenceResult, inferenceTime] = await runSqueezenetModel(
      nodes,
      edges
    );

    // Get the highest confidence.
    var topResult = inferenceResult[0];

    // Update the label and confidence
    setLabel(topResult.name.toUpperCase());
    setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);
  };

  return (
    <>
      <button className={styles.grid} onClick={displayImageAndRunInference}>
        Run Squeezenet inference
      </button>
      <label htmlFor="csvFileSelector" className={styles.InputLabel}>
        Choose Nodes
      </label>
      <input
        type="file"
        id="csvFileSelector"
        className={styles.Input}
        accept={acceptableCSVFileTypes}
        onChange={handleCsvNodes}
      />
      <label htmlFor="csvFileSelector" className={styles.InputLabel}>
        Choose Edges
      </label>
      <input
        type="file"
        id="csvFileSelector"
        className={styles.Input}
        accept={acceptableCSVFileTypes}
        onChange={handleCsvEdges}
      />
      {/* <input type="file" onChange={handleCsvNodes} />
      <input type="file" onChange={handleCsvEdges} /> */}
      {/* <button className={styles.grid} onClick={handleCsvNodes}>
        Input Nodes
      </button> */}
      <br />
      <canvas ref={canvasRef} width={props.width} height={props.height} />
      <span>
        {topResultLabel} {topResultConfidence}
      </span>
      <span>{inferenceTime}</span>
    </>
  );
};

export default ImageCanvas;
