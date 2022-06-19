import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs/promises'
import * as imageDataLibrary from '@andreekeberg/imagedata'

enum ImageKind {
    KEPT = "kept",
    DELETED = "deleted"
}

function getImageTensorFromFilePath(fileUrl: string): tf.Tensor<tf.Rank.R3> {
    // return new Promise((res, rej) => {
    //     imageData.get(fileUrl, async (error, img) => {
    //         if (error) rej(error)
    //         const imageData = {data: new Uint8Array(img.data), width: img.width, height: img.height}
    //         const imageTensor = await tf.browser.fromPixelsAsync(imageData) 
    //         res(imageTensor);
    //     })
    // })

    const img = imageDataLibrary.getSync(fileUrl);
    const imageData = {data: new Uint8Array(img.data), width: img.width, height: img.height}
    const imageTensor = tf.browser.fromPixels(imageData) 
    return imageTensor;
}

async function getAllFiles() {
    const keptFiles = (await fs.readdir(path.join(process.cwd(), "train_assets", "kept"))).filter(el => !el.startsWith("."))
    const deletedFiles = (await fs.readdir(path.join(process.cwd(), "train_assets", "deleted"))).filter(el => !el.startsWith("."))

    return {keptFiles, deletedFiles}
}

async function loadBaseModel() {
    return await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', { fromTFHub: true })
}

function getFeatureFromFilePath(baseModel:tf.GraphModel, fileName: string, imageKind: ImageKind) {
    const keptImageTensorRaw = getImageTensorFromFilePath(path.join("train_assets/"+imageKind,fileName))
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
            index++;
            const kind = Math.floor(Math.random() * 2) === 0 ? ImageKind.KEPT : ImageKind.DELETED;
            if (kind === ImageKind.KEPT) {
                const feature = getFeatureFromFilePath(baseModel, keptFiles[index], ImageKind.KEPT)
                yield {xs: feature.expandDims(), ys: tf.tensor1d([1, 0]).expandDims()};
                feature.dispose()
            } else {
                const feature = getFeatureFromFilePath(baseModel, deletedFiles[index], ImageKind.DELETED)
                yield {xs: feature.expandDims(), ys: tf.tensor1d([0, 1]).expandDims()};
                feature.dispose()
            }
        }
    }
}

async function run() {
    const {keptFiles, deletedFiles} = await getAllFiles()

    const baseModel = await loadBaseModel()

    const ds = tf.data.generator(getDataGenerator(baseModel, keptFiles, deletedFiles));
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

    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [1024], units: 100, activation: "relu"}))
    model.add(tf.layers.dense({units: 2, activation: "softmax"}))

    model.compile({
        optimizer: "adam",
        metrics: ["accuracy"],
        loss: "binaryCrossentropy"
    })

    model.summary()

    await model.fitDataset(ds, {
        epochs: 10,
        batchesPerEpoch: 5
    });

    model.save('file://./kod_model')
  

}
run()