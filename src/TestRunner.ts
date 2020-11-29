// cant use jest to run TfJS tests
// https://stackoverflow.com/questions/57452981/tensorflowjs-save-model-throws-error-unsupported-typedarray-subtype-float32arr

import { TfClassifier, IMatch, ITaggedInput } from "./TfClassifier";
import { readCsvFile } from './FileUtils'
import chalk from 'chalk'
const debug = require('debug-levels')('TestRunner')
// const debug = require('debug-levels')('TestRunner')

// use a cached version of the models
// const useCachedModel = false
const useCachedModel = true

const TestRunner = {

  // async prepare(): Promise<TfClassifier> {
  //   await model.loadEncoder()
  //   return model
  // },

  async trainOne(topic: string): Promise<TfClassifier> {
    const model = new TfClassifier(topic) // should be different in production
    const trainFile = `./data/inputs/${topic}-train.csv`
    const data: ITaggedInput[] = await readCsvFile(trainFile, __dirname)
    await model.trainModel({ data, useCachedModel: useCachedModel })
    return model
  },

  async trainAll() {
    const model = await TestRunner.trainOne('college')
  },

  async testClassifier(model: TfClassifier, topic: string) {
    const fp = `./data/inputs/${topic}-test.csv`
    const testLines = (await readCsvFile(fp)).slice(0, 2)

    // console.log('passed\tactual\texpect\tconfidence\t\ttext')
    testLines.map(async testInput => {
      const matches: IMatch[] | undefined =
        await model.classify(testInput.text, { expand: true })

      debug.log('test', { testInput, matches })

      const first = matches![0]
      console.assert(first.tag === testInput.tag, 'mismatch', first, testInput)

      // console.log('input:', testInput.text)
      // console.log('topMatch', matches![0])
      // console.log('matches', matches)
    })
  },

  async testAndTrain(topic) {
    const model = await TestRunner.trainOne(topic)
    await TestRunner.testClassifier(model, topic)
  },

  async testOne(topic) {
    const model = new TfClassifier(topic) // should be different in production
    // await model.loadEncoder()
    await model.loadCachedModel()
    await TestRunner.testClassifier(model, topic)
  },

  async testSwitchTopic() {
    const model = new TfClassifier('college') // should be different in production
    // await model.loadEncoder()
    await model.loadCachedModel()
    model.setTopic('chat')
    await model.loadCachedModel()
    await TestRunner.testClassifier(model, 'chat')
  },

  // try to load cached model that does not exist
  async testLoadWithNoCache() {
    const model = new TfClassifier('college') // should be different in production
    await model.setTopic('randomNewThing')
    // await model.loadEncoder()
    const useCached = await model.loadCachedModel() ||
      debug.log('failed to get cached model - need to train')
    console.assert(useCached !== true, 'useCached should not work if not cached')
  },

  async runSuite() {
    await TestRunner.testLoadWithNoCache()
    // await TestRunner.testSwitchTopic()
    // await TestRunner.testAndTrain('chat')
    // await TestRunner.testOne('chat')

    // await TestRunner.testOne('college')

    // const model = await TestRunner.trainOne('college')
    // await TestRunner.predict(model, 'college')
    // const model2 = await TestRunner.trainOne('chat')
    // const topic = 'college'
    // const model = new TfClassifier(topic)
    // await model.loadEncoder()
    // await model.loadCachedModel()
    // await model.loadTrainingData()
    // await TestRunner.testClassifier(model, topic)
  }

}

TestRunner.runSuite()

export { TestRunner }
