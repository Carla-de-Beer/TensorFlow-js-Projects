// Carla de Beer
// Created: April 2018.
// Simple regression exercise with P5js and the Tensorflow.js API (based on the Tensorflow.js example).

const rad = 8;

function setup() {
  createCanvas(500, 500);
  background(240);

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd"
  });

  const list1 = [1, 2, 3, 4];
  const list2 = [1, 3, 5, 7];
  // Generate some synthetic data for training.
  const xs = tf.tensor2d(list1, [4, 1]);
  const ys = tf.tensor2d(list2, [4, 1]);

  for (let i = 0; i < 4; ++i) {
    stroke(100);
    strokeWeight(2.5);
    noFill();
    ellipse(list1[i] * 50, list2[i] * 50, rad, rad);
    noStroke();
    fill(100);
    text(i + 1, list1[i] * 50 + 8, list2[i] * 50 + 5);
    text(`(${list1[i]}, ${list2[i]})`, list1[i] * 50 + 20, list2[i] * 50 + 5);
  }

  stroke(100);
  strokeWeight(0.5);
  for (let i = 0; i < 5; ++i) {
    line(list1[i] * 50, list2[i] * 50, list1[i + 1] * 50, list2[i + 1] * 50);
  }
  noStroke();

  let resY = 0;
  // Train the model using the data.
  model.fit(xs, ys).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    let result = model.predict(tf.tensor2d([5], [1, 1]));
    result.print();
    resY = result.dataSync()[0];
    console.log(resY);

    let lastX = list1[3] + 1;
    stroke(200, 0, 0);
    strokeWeight(2.5);
    noFill();
    ellipse(5 * 50, resY * 50, rad, rad);
    noStroke();
    fill(200, 0, 0);
    text(lastX, 5 * 50 + 8, resY * 50 + 5);
    text(`(${lastX}, ${resY})`, lastX * 50 + 20, resY * 50 + 5);
    stroke(100);
    strokeWeight(0.5);
    line(list1[3] * 50, list2[3] * 50, lastX * 50, resY * 50);
    let error = 9 - resY;
    fill(100);
    text(`Error: ${error}`, 20, height - 20);
  });
}