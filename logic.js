let data;
let xs, ys;
let model;
let labelP;
let lossP;
let label;

//list of labels the classifier can choose from
let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

//importing the data used to train the model
function preload() {
  data = loadJSON('https://raw.githubusercontent.com/CodingTrain/ColorClassifer-TensorFlow.js/master/colorData.json');
}

function setup() {
  labelP = createP('');
  lossP = createP('');
  console.log(data.entries.length);
  let colors = [];
  let labels = [];
  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255];
    //pushing color's RGB value to colors array
    colors.push(col);
    //pushing the corresponding labeshing colors arrayl index from the labelList to the labels array
    labels.push(labelList.indexOf(record.label));
  }



  // --passing colors array into tf.tensor2d
  xs = tf.tensor2d(colors);

  let labelsTensor = tf.tensor1d(labels, 'int32');
  labelsTensor.print();


  ys = tf.oneHot(labelsTensor, 9);
  // disposing of used data to prevent lag
  labelsTensor.dispose();

  console.log(xs.shape);
  console.log(ys.shape);

  xs.print();
  ys.print();


  //Creating our machine learning model
  model = tf.sequential();

  //Structuring the input
  let hidden = tf.layers.dense({
    units: 16,
    activation: 'sigmoid',
    inputDim: 3
  });

  //Structuring the output
  let output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  });
  model.add(hidden);
  model.add(output);

  //Creating an optimizer to help train our model
  const lr = 0.2;
  const optimizer = tf.train.sgd(lr);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  });
}

//Training our model, programmed for 10 epochs
function train() {
  const options = {
      epochs: 10,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
          onTrainBegin: () => console.log('training start'),
          onTrainEnd: () => console.log('training complete'),
          onBatchEnd: tf.nextFrame,
          onEpochEnd: (num, logs) => {
              console.log('Epoch: ' + num);
              document.getElementById("loss").innerHTML = logs.loss;
              console.log('Loss: ' + logs)
          }
      }
  }
  return model.fit(xs, ys, options)
}

//
function draw() {
  tf.tidy(() => {
      const xs = tf.tensor2d([
          [r / 255, g / 255, b / 255]
      ]);
      let results = model.predict(xs);
      let index = results.argMax(1).dataSync()[0];

      label = labelList[index];
      document.getElementById("prediction").innerHTML = label;
  });
}




//Listening to value changes on our R, G, B sliders
var rSlider = document.getElementById("rSlider");
var gSlider = document.getElementById("gSlider");
var bSlider = document.getElementById("bSlider");

var rValue = document.getElementById("rValue");
var gValue = document.getElementById("gValue");
var bValue = document.getElementById("bValue");

var r = 127.5;
var g = 127.5;
var b = 127.5;

rValue.innerHTML = rSlider.value;
gValue.innerHTML = gSlider.value;
bValue.innerHTML = bSlider.value;

rSlider.oninput = function () {
  r = this.value;
  rValue.innerHTML = this.value;
  updateCanvas();
}

gSlider.oninput = function () {
  g = this.value;
  gValue.innerHTML = this.value;
  updateCanvas();
}

bSlider.oninput = function () {
  b = this.value;
  bValue.innerHTML = this.value;
  updateCanvas();
}

function updateCanvas() {
  r.toString();
  g.toString();
  b.toString();

  var rgb = "rgb(" + r + ", " + g + ", " + b + ")";

  document.getElementById("canvas").style.backgroundColor = rgb;
  document.getElementById("rgb-value").innerHTML = rgb;

}

updateCanvas();
