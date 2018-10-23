// Carla de Beer
// Created: October 2018.
// Neural network that predicts whether white or black text is better suited
// to a given coloured background.
// Based on Daniel Shiffman's Coding Train video example:
// https://www.youtube.com/watch?v=KtPpoMThKUs
// The code has been amended from example shown through the application of the Tensorflow.js API.

let data;
let model;
let winner;
let xs, ys;
let lossP, labelP, trainingP;
let rSlider, gSlider, bSlider;
let r, g, b;

let labelList = [
  "black", "white"
];

function setup() {
  createCanvas(600, 300);
  lossP = createP("Loss: ");
  labelP = createP("Prediction: ");
  trainingP = createP();

  rSlider = createSlider(0, 255, floor(random(255)));
  gSlider = createSlider(0, 255, floor(random(255)));
  bSlider = createSlider(0, 255, floor(random(255)));

  let colors = [];
  let labels = [];

  for (let i = 0; i < 10000; ++i) {
    let rr = random(255);
    let gg = random(255);
    let bb = random(255);
    colors.push([rr / 255, gg / 255, bb / 255]);

    if (rr + gg + bb > (255 * 3) / 2) {
      labels.push(0);
    } else {
      labels.push(1);
    }
  }

  // console.log(colors);
  // console.log(labels);

  xs = tf.tensor2d(colors);
  const labelsTensor = tf.tensor1d(labels, "int32");
  labelsTensor.print();

  // One-hot encoding
  ys = tf.oneHot(labelsTensor, 2);
  labelsTensor.dispose();

  // console.log(xs.shape);
  // console.log(ys.shape);
  // xs.print();
  // ys.print();

  model = tf.sequential();

  const hidden = tf.layers.dense({
    units: 15,
    activation: "sigmoid",
    inputDim: 3
  });

  // softmax: activation function for generating a probability distribution
  const output = tf.layers.dense({
    units: 2,
    activation: "softmax"
  });

  model.add(hidden);
  model.add(output);

  // Create an optimiser; categoricalCrossEntropy
  const lr = 0.25;
  const optimiser = tf.train.sgd(lr);

  // Compile the model
  // Add loss function (optimise against the loss function)
  // categoricalCrossentropy: loss function for comparing two probability distributions
  model.compile({
    optimizer: optimiser,
    loss: "categoricalCrossentropy"
  });

  // Train the model
  train();
}

// Training Function
async function train() {
  const options = {
    epochs: 2,
    validationSplit: 0.1,
    shuffle: true,
    callbacks: {
      onTrainBegin: () => console.log("Training: started ..."),
      onTrainEnd: () => {
        trainingP.html("Training: completed");
        console.log("Training: completed.")
      },
      onBatchEnd: tf.nextFrame,
      onEpochEnd: (num, logs) => {
        console.log("Epoch: " + num);
        console.log("Loss: " + logs.loss);
        lossP.html("Loss: " + logs.loss);
      }
    }
  }
  return await model.fit(xs, ys, options);
}

function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);

  strokeWeight(10);
  stroke(255);
  line(width / 2 - 5, 0, width / 2 - 5, height);

  textSize(64);
  noStroke();
  textAlign(CENTER, CENTER);
  fill(0);
  text(labelList[0], 150, 60);
  fill(255);
  text(labelList[1], 450, 60);

  tf.tidy(() => {
    const input = tf.tensor2d([
      [r / 255, g / 255, b / 255]
    ]);

    let results = model.predict(input);
    //console.log(results);
    // argMax: returns indices of the maximum values along an axis
    // dataSync: data is retrieved asyncronously
    let index = results.argMax(1).dataSync()[0];
    let label = labelList[index];
    labelP.html(`Prediction: ${label}`);

    let col;
    if (label === labelList[0]) {
      col = 0;
      pos = 150;
      fill(col);
      ellipse(150, 175, 60, 60);
    } else {
      col = 255;
      pos = 450;
      fill(col);
      ellipse(450, 175, 60, 60);
    }

    winner = tf.softmax(results).dataSync()[index];
    console.log(`softmax: ${tf.softmax(results).dataSync()}`);
    console.log(`argMax: ${index}`);

    textSize(12);
    fill(col);
    text(`Degree of certainty: ${winner.toFixed(4)}`, pos, 270);

  });

  tf.memory().numTensors;
}