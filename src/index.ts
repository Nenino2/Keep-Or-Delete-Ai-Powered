import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs/promises'
import * as imageData from '@andreekeberg/imagedata'

enum ImageKind {
    KEPT = "kept",
    DELETED = "deleted"
}

function getImageTensorFromFilePath(fileUrl: string): Promise<tf.Tensor<tf.Rank.R3>> {
    return new Promise((res, rej) => {
        imageData.get(fileUrl, async (error, img) => {
            if (error) rej(error)
            const imageData = {data: new Uint8Array(img.data), width: img.width, height: img.height}
            const imageTensor = await tf.browser.fromPixelsAsync(imageData) 
            res(imageTensor);
        })
    })
}

async function getAllFiles() {
    const keptFiles = (await fs.readdir(path.join(process.cwd(), "assets", "kept"))).filter(el => !el.startsWith("."))
    const deletedFiles = (await fs.readdir(path.join(process.cwd(), "assets", "deleted"))).filter(el => !el.startsWith("."))

    return {keptFiles, deletedFiles}
}

async function loadBaseModel() {
    return await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', { fromTFHub: true })
}

async function getFeatureFromFilePath(baseModel:tf.GraphModel, fileName: string, imageKind: ImageKind) {
    const keptImageTensorRaw = await getImageTensorFromFilePath(path.join("assets/"+imageKind,fileName))
    const keptImageTensor = keptImageTensorRaw.resizeBilinear([224, 224], true).div(255).expandDims()
    const feature = baseModel.predict(keptImageTensor) as tf.Tensor
    const featureSqueezed = feature.squeeze();

    keptImageTensorRaw.dispose()
    keptImageTensor.dispose()
    feature.dispose()

    return featureSqueezed;
}

function getDataGenerator(baseModel: tf.GraphModel, keptFiles: string[], deletedFiles: string[]) {
    return function* dataGenerator() {
        const numElements = 100;
        let index = 0;
        while (index < numElements) {
            const feature = await getFeatureFromFilePath(baseModel, keptFiles[index], ImageKind.KEPT)
            yield feature;
        }
        index = 0
        while (index < numElements) {
            const feature = await getFeatureFromFilePath(baseModel, deletedFiles[index], ImageKind.DELETED)
            yield feature;
        }
    }
}

async function run() {
    const {keptFiles, deletedFiles} = await getAllFiles()

    const baseModel = await loadBaseModel()

    const ds = tf.data.generator(dataGenerator(baseModel, keptFiles, deletedFiles));

    /*

    const features = [];
    const outputs = [];

    for (let i = 0; i<2; i++) {
        const feature = await getFeatureFromFilePath(baseModel, keptFiles[i], ImageKind.KEPT)
        features.push(feature)
        outputs.push([1,0])
    }

    for (let i = 0; i<2; i++) {
        const feature = await getFeatureFromFilePath(baseModel, deletedFiles[i], ImageKind.DELETED)
        features.push(feature)
        outputs.push([0,1])
    }

    console.log(features)
    console.log(outputs)

    */

    /*
    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [1024], units: 100, activation: "relu"}))
    model.add(tf.layers.dense({units: 2, activation: "softmax"}))

    model.compile({
        optimizer: "adam",
        metrics: ["accuracy"],
        loss: "binaryCrossentropy"
    })

    const results = await model.fit(
		tf.stack(features), // input
		tf.stack(outputs), // output
		{
            shuffle: true, // to make sure to randomize the input's order
            batchSize: 1, // how many input to train before changing weights
            epochs: 10 // how many times to run the training on all inputs
        }
    )

    const newSampleImage = await getFeatureFromFilePath(baseModel, deletedFiles[Math.floor(Math.random() * deletedFiles.length - 1)], ImageKind.DELETED)

    const prediction = model.predict(newSampleImage.expandDims()) as tf.Tensor
    
    console.log(await prediction.array())
*/


}

run()