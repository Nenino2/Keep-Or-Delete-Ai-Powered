import * as tf from '@tensorflow/tfjs-node';
import { getAllInputFiles, loadBaseModel, loadModel, predict } from './modules/utils';
import path from "path"
import moveFile from 'move-file';

async function run() {
    const model = await loadModel()
    const baseModel = await loadBaseModel()

    const {inputFiles} = await getAllInputFiles()

    for (let i = 0; i < inputFiles.length; i++) {
            const inputFileName = inputFiles[i];
            const inputFilePath = path.join(process.cwd(), "inputs/all", inputFileName)
            console.log(inputFilePath)

            const prediction = tf.tidy(() =>{
                const prediction = predict(baseModel, model, inputFilePath)
                return prediction;
            })
            await moveFile(inputFilePath, path.join(`output/${prediction}/${inputFileName}`))
    }
}
run()

