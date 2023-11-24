import * as tf from '@tensorflow/tfjs';

(async () => {
    const status = document.getElementById('status');
    if (status) {
      status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
    }
    
})();
