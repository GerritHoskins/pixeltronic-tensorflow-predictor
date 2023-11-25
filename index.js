let bodypix;
let segmentation;
let segmenter;
let imageElement;
let canvas;
let ctx;
let poseNet;
let poses = [];

const width = 713;
const height = 1267;

let bodyPixModel;

function setup() {
    // Create a canvas element and get its context
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Load the image
    imageElement = new Image();
    imageElement.onload = imageLoaded;
    imageElement.src = 'data/sara.jpg'; // Replace with your image path
}

async function imageLoaded() {
    console.log('Image Loaded');
    // Manually set the canvas size
    canvas.width = width;
    canvas.height = height;

    const segmenterConfig = {
      architecture: 'MobileNetV1',
      outputStride: 8,
      multiplier:1, 
      quantBytes: 4
    };

    segmenter = await bodySegmentation.createSegmenter(bodySegmentation.SupportedModels.BodyPix, segmenterConfig);
    const segmentation = await segmenter.segmentPeople(imageElement, {multiSegmentation: false, segmentBodyParts: true, internalResolution: 1, segmentationThreshold: 0.8 });

    // The colored part image is an rgb image with a corresponding color from the
    // rainbow colors for each part at each pixel, and black pixels where there is
    // no part.
    const coloredPartImage = await bodySegmentation.toColoredMask(segmentation, bodySegmentation.bodyPixMaskValueToRainbowColor, {r: 255, g: 255, b: 255, a: 255});
    const opacity = 0.7;
    const flipHorizontal = false;
    const maskBlurAmount = 0;

   // drawBodyPart(bodyPixModel.mask.mask);

   bodySegmentation.drawMask(
    canvas, imageElement, coloredPartImage, opacity, maskBlurAmount,
    flipHorizontal);
    
}

async function modelReady() {
    console.log('Model Ready');
    // Perform segmentation
    //bodyPixModel.segmentWithParts(imageElement, gotResults);
    
}

function drawBodyPart(segmentation) {
    // The specific body part's ID (check BodyPix documentation for IDs)
    const bodyPartId = 12; // Example ID for right arm

    // Draw the body part
    for (let y = 0; y < segmentation.height; y++) {
        for (let x = 0; x < segmentation.width; x++) {
            let index = y * segmentation.width + x;
            if (segmentation.data[index] === bodyPartId) {
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)'; // Red color with some transparency
                ctx.fillRect(x, y, 1, 1); // Drawing each pixel of the body part
            }
        }
    }
}
window.addEventListener('DOMContentLoaded', async () => { 
// Start the sketch
setup();
});
