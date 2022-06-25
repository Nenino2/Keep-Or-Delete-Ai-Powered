import * as tf from "@tensorflow/tfjs"
import { currentImage, uploadButton } from './Dom';

export enum ImageKind {
    KEPT = "kept",
    DELETED = "deleted"
}

export async function loadBaseModel() {
    return await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', { fromTFHub: true })
}

export async function loadModel() {
    return await tf.loadLayersModel('/kod_model/model.json')
}

export function getImageTensorFromFilePath(file: File): Promise<tf.Tensor<tf.Rank.R3>> {
    return new Promise((res, rej) => {
        try {
            const fileReader = new FileReader();
            fileReader.readAsDataURL(file);
            fileReader.onloadend = () => {
                currentImage.onload = () => {
                    const imageTensor = tf.browser.fromPixels(currentImage)
                    res(imageTensor);
                }
                currentImage.src = fileReader.result as string
    
            }
        } catch(error) {
            rej(error)
        }
    });
}

export async function getFeatureFromFilePath(baseModel:tf.GraphModel, file: File) {
    const keptImageTensorRaw = await getImageTensorFromFilePath(file)
    const keptImageTensor = keptImageTensorRaw.resizeBilinear([224, 224], true).div(255).expandDims()
    const feature = baseModel.predict(keptImageTensor) as tf.Tensor
    const featureSqueezed = feature.squeeze();

    keptImageTensorRaw.dispose()
    keptImageTensor.dispose()
    feature.dispose()

    return featureSqueezed;
}

export async function predict(baseModel: tf.GraphModel, model: tf.LayersModel, file: File) {
    const inputImage = await getFeatureFromFilePath(baseModel, file)

    const prediction = model.predict(inputImage.expandDims()) as tf.Tensor

    const result = prediction.squeeze().argMax().arraySync() === 0 ? ImageKind.KEPT : ImageKind.DELETED;

    inputImage.dispose();
    prediction.dispose();

    return result;
}

export async function runAi() {
    try {
        const dirHandle = await (window as any).showDirectoryPicker();

        uploadButton.classList.toggle("invisible");
        currentImage.classList.toggle("invisible")

        const baseModel = await loadBaseModel()
        const model = await loadModel();

        for await (const entry of dirHandle.values()) {
            const file: File = await entry.getFile();
            const result = await predict(baseModel, model, file)
            console.log(result);
            break;
        }
    } catch(error) {
        console.warn("Aborted")
    }
}