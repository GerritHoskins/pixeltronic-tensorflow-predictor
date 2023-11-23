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

async function preprocessImage(img, targetWidth = 224, targetHeight = 224) {
    return tf.tidy(() => {
        // Convert the image to a tensor
        let tensor = tf.browser.fromPixels(img);

        // Resize the image to the target dimensions
        tensor = tf.image.resizeBilinear(tensor, [targetHeight, targetWidth]);

        // Normalize the image from [0, 255] to [0, 1]
        tensor = tensor.toFloat().div(tf.scalar(255));

        // Add the batch dimension
        tensor = tensor.expandDims(0);

        return tensor;
    });
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

function createImagePredictionModel(inputHeight, inputWidth, outputHeight, outputWidth) {
    const model = tf.sequential();

    // Example input layer: assuming the input is an RGB image
    model.add(tf.layers.conv2d({
        inputShape: [inputHeight, inputWidth, 3], // 3 for RGB channels
        filters: 224,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));

    // Add more layers: Convolutional, MaxPooling, Dropout, etc.
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    // Add more layers as needed...

    // Upsampling and Convolutional layers to reach the desired output image size
    // Example: if the output image is 64x64 RGB
    model.add(tf.layers.upSampling2d({size: 2}));
    model.add(tf.layers.conv2d({
        filters: 224,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));

    model.add(tf.layers.upSampling2d({size: 2}));
    model.add(tf.layers.conv2d({
        filters: 3, // 3 filters for RGB output
        kernelSize: 3,
        activation: 'sigmoid', // 'sigmoid' to output values between 0 and 1
        padding: 'same'
    }));

    model.compile({
        optimizer: 'adam', // Adam is a good default choice
        loss: 'meanSquaredError', // For image-to-image, mean squared error can be a good starting point
        metrics: ['accuracy'] // Depending on your task, you might want different metrics
    });
    

    return model;
}


async function trainModel(trainingData, trainingLabels) {
    // Create the model
    const model = tf.sequential();

    const numColorChannels = 3;

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
        units: 2, // Replace with the actual number of output units
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

    const imagePaths = ['data/Off-samples/0.jpg', 'data/Off-samples/1.jpg', 'data/Off-samples/2.jpg', 'data/Off-samples/3.jpg', 
    'data/Off-samples/4.jpg', 'data/Off-samples/5.jpg', 'data/On-samples/0.jpg', 'data/On-samples/1.jpg', 'data/On-samples/2.jpg', 'data/On-samples/3.jpg', 
    'data/On-samples/4.jpg', 'data/On-samples/5.jpg']; // Paths to your images
    const labelData = ["On","Off"]; // Corresponding labels for your images

    const imageTensors = await Promise.all(imagePaths.map(async (path) => {
        const img = await loadImage(path);
        return preprocessImage(img); // Preprocess the image (resize, normalize, etc.)
    }));

    const uniqueLabels = Array.from(new Set(labelData)); // Get unique labels
    const labelIndices = labelData.map(label => uniqueLabels.indexOf(label)); // Convert to indices

    const imagesTensor = tf.stack(imageTensors);
    const labelTensors = tf.tensor1d(labelIndices, 'int32'); // Convert to int32 tensor
    const labelsTensor = tf.oneHot(labelTensors, 2); // Use one-hot encoding for labels

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

    const inputHeight = 224; // Height of input images
    const inputWidth = 224; // Width of input images
    const outputHeight = 224; // Height of output images (predicted)
    const outputWidth = 224; // Width of output images (predicted)

       
    inputElement.addEventListener('change', (e) => {
    imgElement.src = URL.createObjectURL(e.target.files[0]);
    }, false);
    imgElement.onload = function () {
    predict(model, imgElement, 'prediction-canvas');
    };

    try {
        model = await loadModel();
    } catch (error) {
        console.log("Model not found, training a new one");
        const {imagesTensor, labelsTensor} = await loadData();
        model = createImagePredictionModel(inputHeight, inputWidth, outputHeight, outputWidth);

        const reshapedImagesTensor = imagesTensor.reshape([12, 224, 224, 3]);
        const history = await model.fit(reshapedImagesTensor, labelsTensor, {
            epochs: 50, // Number of epochs
            validationSplit: 0.2 // Part of data used for validation
        });
        await model.save('localstorage://my-tattoo-model');
    }
    // Add logic to handle user input and call predict()
})();
