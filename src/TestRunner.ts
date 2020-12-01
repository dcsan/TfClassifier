//x @ts-nocheck

// cant use jest to run TfJS tests
// https://stackoverflow.com/questions/57452981/tensorflowjs-save-model-throws-error-unsupported-typedarray-subtype-float32arr
import { strict as assert } from 'assert';

import {
  TfClassifier, IMatch,
  ITaggedInput,
  ClassifyResult
} from "./TfClassifier";
import { readCsvFile } from './utils/FileUtils'
import { TextUtils } from './utils/TextUtils'

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
      const testRes: ClassifyResult | undefined =
        await model.classify(testInput.text, { expand: true })

      debug.log('test', { testInput, testRes })

      const first = testRes?.matches![0]
      console.assert(first?.tag === testInput.tag, 'mismatch', first, testInput)

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

  stringCompares(s1, s2) {
    debug.log('\n----')
    debug.log('s1', s1)
    debug.log('s2', s2)
    const jwScore = TextUtils.JaroWrinker(s1, s2)
    const bowScore = TextUtils.wordsIntersection(s1, s2)
    debug.log('jwScore:', jwScore)
    debug.log('bowScore:', bowScore)
  },

  checkSimilarity(simFunc) {
    simFunc(
      'The quick brown fox jumped over the lazy dog',
      'The small brown fox jumped from dog to table'
    )

    simFunc(
      'The dog was small',
      'It was a small dog'
    )

    simFunc(
      'Was he expected to win the race?',
      'Did he expect to win?'
    )

    simFunc(
      'Did he plan to win?',
      'Did he expect to win?'
    )

    simFunc(
      'This sentence is short',
      'This is a short sentence'
    )

    simFunc(
      'This sentence is a bit longer with a couple of changes',
      'This sentence is a longer with some changes',
    )

    simFunc(
      'The dog was small',
      'the rabbit saw the dog'
    )

    simFunc(
      'The dog was small',
      'blue fish for all'
    )

    simFunc(
      'The dog was small',
      'blue fish random things'
    )

    simFunc(
      'The dog was small',
      'the rabbit saw the dog but there was much else to see'
    )

    simFunc(
      'The dog was small',
      'It was a small dog but there was much else to see'
    )

  },

  // test values recorded in metadata are matching
  async testMeta() {
    const klassy = await TestRunner.trainOne('college')
    const result1 = await klassy.classify('Read the news in English Language')
    // const result2 = await klassy.classify('Read English Language news')
    // debug.log('result1', result1)
    // debug.log('result2', result2)
    const meta = result1!.meta!
    assert(meta.confRatio > 2, 'meta confRatio')  // important > 1.0 !
    assert(meta.jwScore > 0.5, 'meta jwScore problem')
    assert(meta.wordsIntersection > 0.2, 'meta wordsIntersection mismatch')
    assert(meta.pct > 48, 'meta wordsIntersection mismatch')
    assert(meta.delta1 > 0.3, `meta delta1 ${meta.delta1}`)
  },

  async runSuite() {
    await this.testMeta()
    // this.checkSimilarity(this.stringCompares)
    // await TestRunner.testLoadWithNoCache()
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
