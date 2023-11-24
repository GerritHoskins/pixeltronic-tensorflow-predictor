let mobilenet = undefined;
let gatherDataState = "STOP_DATA_GATHER";
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

function enableCam() {
  // TODO: Fill this out later in the codelab!
}


function trainAndPredict() {
  // TODO: Fill this out later in the codelab!
}


function reset() {
  // TODO: Fill this out later in the codelab!
}



function gatherDataForClass() {
  // TODO: Fill this out later in the codelab!
}

async function loadMobileNetFeatureModel() {
  const URL = 
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

(async () => {    
    // Call the function immediately to start loading.
    loadMobileNetFeatureModel();

    const STATUS = document.getElementById('status');
    const VIDEO = document.getElementById('webcam');
    const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
    const RESET_BUTTON = document.getElementById('reset');
    const TRAIN_BUTTON = document.getElementById('train');
    const MOBILE_NET_INPUT_WIDTH = 224;
    const MOBILE_NET_INPUT_HEIGHT = 224;
    const STOP_DATA_GATHER = -1;
    const CLASS_NAMES = [];

    ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
    TRAIN_BUTTON.addEventListener('click', trainAndPredict);
    RESET_BUTTON.addEventListener('click', reset);

    let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
    for (let i = 0; i < dataCollectorButtons.length; i++) {
      dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
      dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
      // Populate the human readable names for classes.
      CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
    }


    const status = document.getElementById('status');
    if (status) {
      status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
    }
    
})();

