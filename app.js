// TensorFlow.js is available as tf

async function loadModel() {
    try {
        // Replace 'model-url' with the URL where your model is hosted
        const model = await tf.loadLayersModel('https://gerrithoskins.github.io/pixeltronic-tensorflow-predictor/models/model1.json');
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

async function trainModel(trainingData, trainingLabels) {
    // Create the model
    const model = tf.sequential();

    // Define the model architecture
    model.add(tf.layers.conv2d({
        inputShape: [224, 224, numColorChannels], // Replace with actual values
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    // Add more layers as needed...

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));

    // Output layer - Adjust units and activation based on your specific case
    model.add(tf.layers.dense({
        units: numOutputUnits, // Replace with the actual number of output units
        activation: 'softmax' // or another appropriate activation function
    }));

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy', // Choose an appropriate loss function
        metrics: ['accuracy'],
    });

    // Train the model
    const history = await model.fit(trainingData, trainingLabels, {
        epochs: 50, // Adjust number of epochs as necessary
        validationSplit: 0.2, // Adjust validation split as necessary
    });

    return model;
}


async function loadData() {
    // Example: Loading image data

    const imagePaths = ['data/Off-samples/1.jpg', 'data/Off-samples/2.jpg', 'data/Off-samples/3.jpg', 
    'data/Off-samples/4.jpg', 'data/Off-samples/5.jpg', 'data/On-samples/1.jpg', 'data/On-samples/2.jpg', 'data/On-samples/3.jpg', 
    'data/On-samples/4.jpg', 'data/On-samples/5.jpg']; // Paths to your images
    const labelData = ["On","Off"]; // Corresponding labels for your images

    const imageTensors = await Promise.all(imagePaths.map(async (path) => {
        const img = await loadImage(path);
        return preprocessImage(img); // Preprocess the image (resize, normalize, etc.)
    }));

    const labelTensors = tf.tensor(labelData); // Convert labels to tensor

    // Combine the images into one tensor and the labels into another tensor
     const imagesTensor = tf.stack(imageTensors);
    //const labelsTensor = tf.oneHot(labelTensors, 2); // Use one-hot encoding for labels if it's a classification task
    
    const labelsTensor = tf.oneHot(labelTensors, numClasses); // Use one-hot encoding for labels


    return {imagesTensor, labelsTensor};
}

async function loadImage(path) {
    return new Promise((resolve, reject) => {
        // Logic to load an image from the path
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = path;
    });
}

function preprocessImage(img) {
    // Convert the image to a tensor and preprocess it
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([224, 224]) // Resize the image
                    .toFloat()
                    .div(tf.scalar(255)); // Normalize the image

        // Additional preprocessing like mean subtraction, etc., can go here

        return tensor.expandDims(); // Add batch dimension
    });
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
        const {imagesTensor, labelsTensor} = await loadData();
        const model = await trainModel(imagesTensor, labelsTensor);
    }

    // Add logic to handle user input and call predict()
})();
