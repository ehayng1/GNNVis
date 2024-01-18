import logo from "./logo.svg";
import "./App.css";
import * as ort from "onnxruntime-web";

import { Tensor, InferenceSession } from "onnxruntime-web";
// import { Tensor, InferenceSession } from "onnxjs";

function App() {
  async function test() {
    const session = new InferenceSession();
    // use the following in an async method
    // const url = "./onnx_model.onnx";
    // const url = "../resgatedgraphconv_Opset16.onnx";
    const url = "./src/resgatedgraphconv_Opset16.onnx";
    console.log("Loading model...");
    await session.loadModel(url);
    console.log("Model loaded successfully!");
    const inputs = [
      new Tensor(
        new Float32Array([
          Math.random(),
          Math.random(),
          Math.random(),
          Math.random(),
        ]),
        "float32",
        [4, 7]
      ),
    ];

    // run this in an async method:
    console.log("Running inference...");
    const outputMap = await session.run(inputs);
    console.log("Inference completed!");

    const outputTensor = outputMap.values().next().value;
    console.log("Output Tensor: ", outputTensor.data);
  }
  async function run() {
    console.log("Inference session initiated");
    const session = await ort.InferenceSession.create(
      // "./resgatedgraphconv_Opset16.onnx",
      "./src/resgatedgraphconv_Opset16.onnx",
      { executionProviders: ["webgl"], graphOptimizationLevel: "all" }
    );

    console.log("Inference session created");

    // Convert input data to ONNX tensor
    const inputTensor = new ort.Tensor(
      new Float32Array([
        Math.random(),
        Math.random(),
        Math.random(),
        Math.random(),
      ]),
      "float32",
      [4, 7]
    );
    console.log("Input Tensor:", inputTensor);

    // Run the model
    const outputMap = await session.run({ input: inputTensor });
    console.log("Output Map:", outputMap);

    // @ts-ignore
    const outputTensor = outputMap.values().next().value;
    console.log("Output Tensor:", outputTensor);
    // const outputDataArray = outputTensor.data;
  }
  // test();
  run();
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
