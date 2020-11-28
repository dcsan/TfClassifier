import * as tf from "@tensorflow/tfjs-node";
import * as fs from 'fs'

// import {
//   // ActivationIdentifier,
//   ActivationSerialization
// } from '@tensorflow/tfjs-node'
// import { DenseLayerArgs } from '@tensorflow/tfjs-node'

import * as sentenceEncoder from "@tensorflow-models/universal-sentence-encoder";
const debug = require('debug-levels')('TfClassifier')
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

export interface ITrainOptions {
  useCachedModel?: boolean
  // FIXME - types ActivationSerialization - https://github.com/tensorflow/tfjs-layers/blob/master/tfjs-layers/src/keras_format/activation_config.ts#L30-L31
  activation?: any,
  epochs?: number
  data?: ITaggedInput[]
  learningRate?: number
  verbose?: number
}

const defaultTrainOptions: ITrainOptions = {
  useCachedModel: false,
  activation: "softmax",
  epochs: 150,
  learningRate: 0.001,   // for adam classifier
  verbose: 0
}

export interface ILoadOpts {
  useCachedModel: boolean
}

const showEndTime = (msg, startTime) => {
  const hrend = process.hrtime(startTime)
  debug.info(`${msg} %ds %dms`, hrend[0], hrend[1] / 1000000)
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
  encoder: any
  modelPath?: string
  modelDir?: string
  modelUrl?: string
  topicName?: string  // eg topic
  model: any
  loaded: boolean = false
  uniqueTags: string[] = []
  trainingData?: ITaggedInput[]

  constructor(topicName = 'tfModel') {
    this.setTopic(topicName)
  }

  // allows to switch to a different topic
  // but keep same loaded encoder
  setTopic(topicName: string) {
    this.topicName = topicName
    this.modelDir = path.join(__dirname, 'data', 'modelCache')
    ensureDirectory(this.modelDir)
    this.modelPath = path.join(this.modelDir, topicName)
    this.modelUrl = `file://${this.modelPath}`
  }

  async loadCachedModel(opts: ILoadOpts = { useCachedModel: true }) {
    await this.loadEncoder()  // always needed
    try {
      const modelFile = `${this.modelUrl}/model.json`
      const loadedModel = await tf.loadLayersModel(
        modelFile
      );
      // TODO - check shape matches data
      debug.log("Using cached model", this.topicName);
      this.model = loadedModel
      await this.loadTrainingData()
      return loadedModel;
    } catch (err) {
      debug.warn("cannot find cached model", this.modelPath, err);
      return false
    }
  }

  // save to local file so we can load with cached model
  async saveTrainingData(trainingData: any[]) {
    const jsonPath = path.join(this.modelPath!, 'trainingData.json')
    try {
      let data = JSON.stringify(trainingData, null, 2)
      fs.writeFileSync(jsonPath, data)
    } catch (err) {
      debug.error('failed to write data', jsonPath)
      throw (err)
    }
  }

  // save to local file so we can load with cached model
  async loadTrainingData() {
    const jsonPath = path.join(this.modelPath!, 'trainingData.json')
    try {
      let rawdata = fs.readFileSync(jsonPath)
      const trainingData = JSON.parse(String(rawdata))
      await this.prepareTags(trainingData)
      debug.log('loaded cached trainingData:', trainingData)
      this.trainingData = trainingData
    } catch (err) {
      debug.error('failed to load trainingData', this.modelPath)
      throw (err)
    }
  }

  prepareTags(trainingData) {
    const allTags: string[] = trainingData.map(t => t.tag)
    this.uniqueTags = _.uniq(allTags)
    return this.uniqueTags
  }

  // only needed for training?
  async loadEncoder() {
    if (this.loaded) return
    var startTime = process.hrtime()
    this.encoder = await sentenceEncoder.load()
    showEndTime('loaded encoder', startTime)
    this.loaded = true
  }

  async encodeData(tasks: any) {
    const sentences = tasks.map(t => t.text.toLowerCase());
    const embeddings = await this.encoder.embed(sentences);
    return embeddings;
  }

  async loadCsvInputs(relPath, basePath = __dirname) {
    this.trainingData = await readCsvFile(relPath, basePath)
    const before = this.trainingData.length
    // filter items without required fields
    this.trainingData = this.trainingData.filter(item => item.tag && item.text)
    const diff = this.trainingData.length - before
    if (diff !== 0) {
      debug.warn('trimmed some items from inputs', diff)
    }
  }

  // force = ignore cached model
  async trainModel(trainParams: ITrainOptions): Promise<tf.Sequential | tf.LayersModel> {

    const trainOpts = Object.assign(defaultTrainOptions, trainParams)

    const trainingData: ITaggedInput[] = trainOpts.data || this.trainingData!
    if (!trainingData || !trainingData.length) {
      throw ('no trainingData for trainModel')
    }
    this.trainingData = trainingData
    await this.prepareTags(trainingData)
    debug.log('trainingData.length', trainingData.length)

    await this.loadEncoder()

    var startTime = process.hrtime()
    if (trainOpts.useCachedModel) {
      this.model = await this.loadCachedModel()
      showEndTime('loaded model', startTime)
    }
    const xTrain = await this.encodeData(trainingData);
    showEndTime('encoded data', startTime)

    // returns an array like [0,0,1,0] for each entry
    const labels = (
      trainingData.map((utt: ITaggedInput) => {
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
        activation: trainOpts.activation,
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

    debug.log('compile')
    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(trainOpts.learningRate),
      metrics: ["accuracy"]
    });

    // const lossContainer = document.getElementById("loss-cont");
    debug.log('fit')
    await model.fit(xTrain, yTrain, {
      batchSize: 32,
      validationSplit: 0.1,
      shuffle: true,
      epochs: trainOpts.epochs,
      verbose: trainOpts.verbose,
      // callbacks: tfvis.show.fitCallbacks(
      //   lossContainer,
      //   ["loss", "val_loss", "acc", "val_acc"],
      //   {
      //     callbacks: ["onEpochEnd"]
      //   }
      // )
    });

    showEndTime('fitted model', startTime)
    this.model = model

    debug.log('writing model', this.modelUrl)
    await model.save(this.modelUrl!)
    await this.saveTrainingData(trainingData)
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
  // note -this assumes all trainingData is in memory
  // which might not be the case for reloading a cached model
  matchingSources(tag: string): ITaggedInput[] | undefined {
    if (!this.trainingData) {
      debug.error('no trainingData so cannot expand sources - are you reloading a model?')
    }
    const sources = this.trainingData?.filter(item => item.tag === tag)
    // debug.log('sources', tag, sources)
    return sources
  }

}

export { TfClassifier };
