let data;
let xs, ys;
let model;
let labelP;
let lossP;
let rSlider, gSlider, bSlider;

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

function preload() {
    data = loadJSON('https://raw.githubusercontent.com/CodingTrain/ColorClassifer-TensorFlow.js/master/colorData.json');
}

function setup() {
    labelP = createP('');
    lossP = createP('loss');
    rSlider = createSlider(0, 255, 255);
    gSlider = createSlider(0, 255, 255);
    bSlider = createSlider(0, 255, 0);

    console.log(data.entries.length);

    let colors = [];
    let labels = [];
    for (let record of data.entries) {
        let col = [record.r / 255, record.g / 255, record.b / 255];
        //pushing RGB values to array
        colors.push(col);
        //pushing RGB labels to array with as index value
        labels.push(labelList.indexOf(record.label));
    }
    // creating a 2d tensor
    xs = tf.tensor2d(colors);

    let labelsTensor = tf.tensor1d(labels, 'int32');
    labelsTensor.print();

    ys = tf.oneHot(labelsTensor, 9);
    labelsTensor.dispose();

    console.log(xs.shape);
    console.log(ys.shape);

    xs.print();
    ys.print();

    model = tf.sequential();

    let hidden = tf.layers.dense({
        units: 16,
        activation: 'sigmoid',
        inputDim: 3
    });
    let output = tf.layers.dense({
        units: 9,
        activation: 'softmax'
    });
    model.add(hidden);
    model.add(output);

    //create optimizer
    const lr = 0.2;
    const optimizer = tf.train.sgd(lr);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    });

    train().then(results => {
        console.log(results.history.loss);
    });
}

async function train() {
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

                lossP.html('Loss: ' + logs.loss)

                console.log('Loss: ' + logs)
            }
        }
    }
    return await model.fit(xs, ys, options)
}

function draw() {
    let r = rSlider.value();
    let g = gSlider.value();
    let b = bSlider.value();

    background(r, g, b);

    tf.tidy(() => {
        const xs = tf.tensor2d([
            [r / 255, g / 255, b / 255]
        ]);
        let results = model.predict(xs);
        let index = results.argMax(1).dataSync()[0];

        let label = labelList[index];
        labelP.html(label);
    });
}