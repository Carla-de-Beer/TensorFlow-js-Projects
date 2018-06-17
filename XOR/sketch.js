// Carla de Beer
// June 2018
// A simple exercise with P5js and the Tensorflow.js API to classify the XOR problem.
// Based on Daniel Shiffman's Coding Train video example:
// https://www.youtube.com/watch?v=N3ZnNa01BPM

let model = {};
let inputs = [];
let xs = [];
let resolution = 20;
let learningRate = 0.1;
let numEpochs = 100;
let cols;
let rows;
// Training data for XOR
const train_xs = tf.tensor2d([[0, 0], [1, 0], [0, 1], [1, 1]]);
// Labels for XOR
const train_ys = tf.tensor2d([[0], [1], [1], [0]]);

function setup() {
  createCanvas(600, 600);
  cols = width / resolution;
  rows = height / resolution;

  // Create the input data
  for (let i = 0; i < cols; ++i) {
    for (let j = 0; j < rows; ++j) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }

  xs = tf.tensor2d(inputs);

  // 1. Create the model (once) with the Tensorflow.js Layers API
  model = tf.sequential();
  let hidden = tf.layers.dense({
    inputShape: [2],
    units: 4,
    activation: "sigmoid"
  });
  let output = tf.layers.dense({
    units: 1
  });
  model.add(hidden);
  model.add(output);

  const sgdOpt = tf.train.adamax(learningRate);
  model.compile({
    optimizer: sgdOpt,
    loss: "meanSquaredError"
  });

  // 2. Train the model (once)
  setTimeout(train, 100);

}

function train() {
  trainModel().then(/*(result) => console.log(result.history.loss[0])*/);
  setTimeout(train, 100);
}

function trainModel() {
  return model.fit(train_xs, train_ys, {
    shuffle: true,
    epochs: numEpochs
  });
}

function draw() {
  background(200, 210, 35);

  tf.tidy(() => {
    // Get the predictions
    let ys = model.predict(xs).dataSync();
    //console.log(ys);

    // Draw the results
    let index = 0;
    for (let i = 0; i < cols; ++i) {
      for (let j = 0; j < rows; ++j) {
        fill(255 - ys[index] * 255, ys[index] * 255, 255 - ys[index] * 155);
        strokeWeight(0.25);
        stroke(255);
        rect(i * resolution, j * resolution, resolution, resolution);
        noStroke();
        fill(255);
        textSize(8);
        textAlign(CENTER, CENTER);
        text(nf(ys[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2);
        index++;
      }
    }
  });
}