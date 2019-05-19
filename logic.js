// window.addEventListener('DOMContentLoaded', (event) => {
//     console.log('DOM fully loaded and parsed');

//     const data = loadJSON('https://raw.githubusercontent.com/CodingTrain/ColorClassifer-TensorFlow.js/master/colorData.json');
// });




var rSlider = document.getElementById("rSlider");
var gSlider = document.getElementById("gSlider");
var bSlider = document.getElementById("bSlider");

var rValue = document.getElementById("rValue");
var gValue = document.getElementById("gValue");
var bValue = document.getElementById("bValue");

rValue.innerHTML = rSlider.value;
gValue.innerHTML = gSlider.value;
bValue.innerHTML = bSlider.value;


rSlider.oninput = function() {
  rValue.innerHTML = this.value;
}

gSlider.oninput = function() {
    gValue.innerHTML = this.value;
  }

bSlider.oninput = function() {
    bValue.innerHTML = this.value;
  }