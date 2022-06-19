import * as tf from '@tensorflow/tfjs-node';
import { getAllTrainingFiles, loadBaseModel, getDataGenerator } from './modules/utils';

async function run() {
    const {keptFiles, deletedFiles} = await getAllTrainingFiles()

    const baseModel = await loadBaseModel()

    const ds = tf.data.generator(getDataGenerator(baseModel, keptFiles, deletedFiles));

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
        batchesPerEpoch: 4
    });

    model.save('file://./kod_model')

}
run()

