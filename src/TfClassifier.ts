// require('@tensorflow/tfjs-node')
import * as tf from "@tensorflow/tfjs-node";
import * as sentenceEncoder from "@tensorflow-models/universal-sentence-encoder";
const debug = require('debug-levels')('TfModel')
import * as _ from 'lodash'
import { readCsvFile } from './FileUtils'

import * as path from 'path'
import { ensureDirectory } from './FileUtils'
// total different categories

export interface ITaggedInput {
  text: string
  tag: string
}

export interface IClassification {
  input: string
  tag: string
  confidence: number
  found?: ITaggedInput
  others: ITaggedInput[]
}

export interface IMatch {
  tag: string
  confidence: number
  pct: number
  sources?: ITaggedInput[]
}

// export type IMatch = [string, number];

// export interface IMatch {
//   tag: string
//   confidence: number
// }
// // {
// //   [string, number]
// // }

class TfClassifier {
  modelPath: string
  modelUrl: string
  encoder: any
  model: any
  loaded: boolean = false
  uniqueTags: string[] = []
  inputs?: ITaggedInput[]

  constructor(modelName = 'tfModel') {
    const modelDir = path.join(__dirname, 'data', 'modelCache')
    ensureDirectory(modelDir)
    this.modelPath = path.join(modelDir, modelName)
    this.modelUrl = `file://${this.modelPath}`
  }

  async loadEncoder() {
    if (this.loaded) return
    this.encoder = await sentenceEncoder.load()
    this.loaded = true
  }

  async encodeData(tasks: any) {
    const sentences = tasks.map(t => t.text.toLowerCase());
    const embeddings = await this.encoder.embed(sentences);
    return embeddings;
  }

  async loadCsvInputs(relPath, basePath = __dirname) {
    this.inputs = await readCsvFile(relPath, basePath)
    const before = this.inputs.length
    // filter items without required fields
    this.inputs = this.inputs.filter(item => item.tag && item.text)
    const diff = this.inputs.length - before
    if (diff !== 0) {
      debug.warn('trimmed some items from inputs', diff)
    }
  }

  // force = ignore cached model
  async trainModel(opts: {
    inputs?: ITaggedInput[],
    useCache: boolean
  }): Promise<tf.Sequential | tf.LayersModel> {

    const inputs: ITaggedInput[] = opts.inputs || this.inputs!
    if (!inputs) {
      throw ('no inputs for trainModel')
    }

    await this.loadEncoder()

    const allTags: string[] = inputs.map(t => t.tag)
    this.uniqueTags = _.uniq(allTags)

    if (opts.useCache) {
      try {
        const modelFile = `${this.modelUrl}/model.json` // annoying TF glitch
        const loadedModel = await tf.loadLayersModel(
          modelFile
        );
        debug.log("Using existing model");
        this.model = loadedModel
        return loadedModel;
      } catch (err) {
        debug.log("err loading model", err);
        debug.log("Training new model");
      }
    }
    const xTrain = await this.encodeData(inputs);

    // returns an array like [0,0,1,0] for each entry
    const labels = (
      inputs.map((utt: ITaggedInput) => {
        const pos = this.uniqueTags.indexOf(utt.tag)
        const mat = new Array(this.uniqueTags.length).fill(0)
        mat[pos] = 1
        return mat
      })
    );
    const yTrain = tf.tensor2d(labels)
    const inputShape = [xTrain.shape[1]]
    const model: tf.Sequential = tf.sequential();

    // debug.log('labels', labels)
    // debug.log('yTrain', {
    //   tags: this.uniqueTags,
    //   labels,
    //   xTrain, yTrain, inputShape
    // })
    // debug.log('xTrain', xTrain)

    model.add(
      tf.layers.dense({
        inputShape: inputShape,
        activation: "softmax",
        units: this.uniqueTags.length // number of classes for classifier
      })
    );

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [xTrain.shape[1]],
    //     activation: "softmax",
    //     units: N_CLASSES
    //   })
    // );
    // model.add(
    //   tf.layers.dense({
    //     inputShape: [xTrain.shape[1]],
    //     activation: "softmax",
    //     units: N_CLASSES
    //   })
    // );

    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(0.001),
      metrics: ["accuracy"]
    });

    // const lossContainer = document.getElementById("loss-cont");

    await model.fit(xTrain, yTrain, {
      batchSize: 32,
      validationSplit: 0.1,
      shuffle: true,
      epochs: 150,
      // callbacks: tfvis.show.fitCallbacks(
      //   lossContainer,
      //   ["loss", "val_loss", "acc", "val_acc"],
      //   {
      //     callbacks: ["onEpochEnd"]
      //   }
      // )
    });

    this.model = model

    debug.log('writing model', this.modelUrl)
    await model.save(this.modelUrl);

    return model;
  }

  async classify(
    input: string,
    opts: { maxHits?: number, expand?: boolean, context?: string } = { maxHits: 10, expand: true }): Promise<IMatch[] | undefined> {
    // const maxHits = opts.maxHits || 10
    input = input.trim()
    if (!input) {
      debug.warn('empty input to predictor')
      return
    }
    const xPredict = await this.encodeData([{ text: input }])
    const tensor = await this.model.predict(xPredict);
    const confidenceList: number[] = await tensor.data()

    // const pcts = pdata.map(p => Math.round(p * 100))
    // find most confident item
    // const maxConfidence = Math.max(...confidenceList)
    // const tagIndex = confidenceList.indexOf(maxConfidence)
    // const tag = this.uniqueTags[tagIndex]

    // debug.log('classification', {
    //   confidenceList,
    //   maxConfidence,
    //   tagIndex,
    //   tag
    // })

    // const vals = pcts.join(',')
    const matches: IMatch[] = this.uniqueTags.map((tag, index) => {
      const confidence = confidenceList[index]
      return {
        confidence,
        tag,
        pct: Math.round(confidence * 100)
      }
    })

    const sortedMatches = matches.sort((a, b) => {
      return (a.confidence < b.confidence ? 1 : -1)
    })

    const topMatches = opts.maxHits ? sortedMatches.slice(0, opts.maxHits) : sortedMatches
    const expMatches = opts.expand ? this.expandMatches(topMatches) : topMatches
    // TODO - filter on context of expanded matches
    return expMatches
  };

  expandMatches(matches: IMatch[]) {
    const expanded = matches.map(m => {
      m.sources = this.matchingSources(m.tag)
      return m
    })
    return expanded
  }

  // find all original training sentences for that tag
  matchingSources(tag: string): ITaggedInput[] | undefined {
    return this.inputs?.filter(item => item.tag === tag)
  }

}

export { TfClassifier };
