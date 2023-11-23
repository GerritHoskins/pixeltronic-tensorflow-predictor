// TensorFlow.js is available as tf

async function loadModel() {
    try {
        // Replace 'model-url' with the URL where your model is hosted
        const model = await tf.loadLayersModel('https://gerrithoskins.github.io/pixeltronic-tensorflow-predictor/models/model.json');
        console.log("Model loaded successfully");
        return model;
    } catch (error) {
        console.error("Error loading the model: ", error);
        throw error;
    }
}

async function preprocessImage(image) {
    // Preprocess the image to fit the input shape of the model
    // This typically involves resizing the image, normalizing values, etc.
    // Example:
     return tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
             .toFloat()
             .div(tf.scalar(255.0))
            .expandDims();
}

function tensorToImage(tensor) {
    // Convert the tensor to an image format
    // The specific implementation depends on the tensor format
    // For example, using a canvas to draw the image
    const [width, height] = tensor.shape.slice(1, 3);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Convert the tensor to imageData
    const imageData = new ImageData(new Uint8ClampedArray(tensor.dataSync()), width, height);
    ctx.putImageData(imageData, 0, 0);

    return canvas;
}

function tensorToCanvas(tensor, canvasId) {
    // Check if tensor has at least 2 dimensions
    if (tensor.shape.length < 2) {
        throw new Error("Tensor does not have enough dimensions (expected at least 2)");
    }

    // Extract width and height from tensor shape
    const [height, width] = tensor.shape.slice(0, 2);

    // Check if width and height are valid numbers
    if (!width || !height) {
        throw new Error(`Invalid tensor shape: width or height is zero or not a number (width: ${width}, height: ${height})`);
    }

    const canvas = document.getElementById(canvasId) || document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Normalize the tensor to 0-255 and reshape for RGBA format
    let tensorData = tensor.mul(255).toInt().dataSync();

    // Handling cases where the tensor is not in RGBA format
    if (tensorData.length !== height * width * 4) {
        // Assuming the tensor is in RGB format, add an Alpha channel
        const tensorWithAlpha = [];
        for (let i = 0; i < tensorData.length; i += 3) {
            tensorWithAlpha.push(...tensorData.slice(i, i + 3), 255); // Adding alpha value 255 (fully opaque)
        }
        tensorData = tensorWithAlpha;
    }

    const imageData = new ImageData(new Uint8ClampedArray(tensorData), width, height);
    ctx.putImageData(imageData, 0, 0);

    return canvas;
}



async function predict(model, imageElement, canvasId) {
    // Preprocess the input image
    const preprocessedInput = await preprocessImage(imageElement);

    // Make a prediction
    const prediction = model.predict(preprocessedInput);

    // Ensure the prediction is complete before converting to canvas
    await prediction.data();

    // Convert the prediction tensor to a canvas and append to the DOM
    const predictedCanvas = tensorToCanvas(prediction.squeeze(), canvasId);

    // Append the canvas to the document
    document.body.appendChild(predictedCanvas);
}

async function trainModel() {
    // Create the model
    const model = tf.sequential();
    // Add layers
    // ...

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // Load and preprocess your data
    const {trainingData, trainingLabels} = await loadData();

    // Train the model
    const history = await model.fit(trainingData, trainingLabels, {
        epochs: 50,
        validationSplit: 0.2,
    });

    return model;
}

async function loadData() {
    // Load and preprocess your dataset
    // ...
    return {trainingData, trainingLabels};
}


// Main logic
(async () => {
    let model;
    let imgElement = document.getElementById('imageSrc');
    let inputElement = document.getElementById('fileInput');
    inputElement.addEventListener('change', (e) => {
      imgElement.src = URL.createObjectURL(e.target.files[0]);
    }, false);
    imgElement.onload = function () {
        predict(model, imgElement);
      };

    try {
        model = await loadModel();
    } catch (error) {
        console.log("Model not found, training a new one");
        model = await trainModel();
    }

    // Add logic to handle user input and call predict()
})();
