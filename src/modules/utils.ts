import * as tf from '@tensorflow/tfjs-node';
import * as imageDataLibrary from '@andreekeberg/imagedata'
import path from "path";
import fs from 'fs/promises'

export enum ImageKind {
    KEPT = "kept",
    DELETED = "deleted"
}

export async function loadBaseModel() {
    return await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', { fromTFHub: true })
}

export function getImageTensorFromFilePath(fileUrl: string): tf.Tensor<tf.Rank.R3> {
    const img = imageDataLibrary.getSync(fileUrl);
    const imageData = {data: new Uint8Array(img.data), width: img.width, height: img.height}
    const imageTensor = tf.browser.fromPixels(imageData) 
    return imageTensor;
}

export function getFeatureFromFilePath(baseModel:tf.GraphModel, filePath: string) {
    const keptImageTensorRaw = getImageTensorFromFilePath(path.join(filePath))
    const keptImageTensor = keptImageTensorRaw.resizeBilinear([224, 224], true).div(255).expandDims()
    const feature = baseModel.predict(keptImageTensor) as tf.Tensor
    const featureSqueezed = feature.squeeze();

    keptImageTensorRaw.dispose()
    keptImageTensor.dispose()
    feature.dispose()

    return featureSqueezed;
}

export function getDataGenerator(baseModel: tf.GraphModel, keptFiles: string[], deletedFiles: string[]) {
    return function* dataGenerator() {
        const numElements = 100;
        let index = 0;
        while (index < numElements) {
            index++;
            const kind = Math.floor(Math.random() * 2) === 0 ? ImageKind.KEPT : ImageKind.DELETED;
            if (kind === ImageKind.KEPT) {
                const feature = getFeatureFromFilePath(baseModel, path.join("train_assets", kind, keptFiles[index] ))
                yield {xs: feature.expandDims(), ys: tf.tensor1d([1, 0]).expandDims()};
                feature.dispose()
            } else {
                const feature = getFeatureFromFilePath(baseModel, path.join("train_assets", kind, deletedFiles[index] ))
                yield {xs: feature.expandDims(), ys: tf.tensor1d([0, 1]).expandDims()};
                feature.dispose()
            }
            console.log("----")
        }
    }
}

export async function getAllTrainingFiles() {
    const keptFiles = (await fs.readdir(path.join(process.cwd(), "train_assets", "kept"))).filter(el => !el.startsWith("."))
    const deletedFiles = (await fs.readdir(path.join(process.cwd(), "train_assets", "deleted"))).filter(el => !el.startsWith("."))

    return {keptFiles, deletedFiles}
}

export function predict(baseModel: tf.GraphModel, model: tf.LayersModel, filePath: string) {
    const inputImage = getFeatureFromFilePath(baseModel, filePath)

    const prediction = model.predict(inputImage.expandDims()) as tf.Tensor

    console.log(prediction.arraySync())
}