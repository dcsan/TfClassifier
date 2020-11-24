// cant use jest to run TfJS tests
// https://stackoverflow.com/questions/57452981/tensorflowjs-save-model-throws-error-unsupported-typedarray-subtype-float32arr

import { TfClassifier, IMatch, ITaggedInput } from "./TfClassifier";
import { readCsvFile } from './FileUtils'
import chalk from 'chalk'
const debug = require('debug-levels')('TestRunner')
// const debug = require('debug-levels')('TestRunner')

// use a cached version of the models
// const useCache = false
const useCache = true

const TestRunner = {

  async prepare(): Promise<TfClassifier> {
    const model = new TfClassifier('testModel') // should be different in production
    await model.loadEncoder()
    return model
  },

  async train(): Promise<TfClassifier> {
    const model = await TestRunner.prepare()
    const data: ITaggedInput[] = await readCsvFile('./data/inputs/train.csv', __dirname)
    await model.trainModel({ data, useCache: useCache })
    return model
  },

  async predict(model: TfClassifier) {
    const testLines = (await readCsvFile('./data/inputs/test.csv')).slice(0, 2)

    // console.log('passed\tactual\texpect\tconfidence\t\ttext')
    testLines.map(async testInput => {
      const matches: IMatch[] | undefined =
        await model.classify(testInput.text, { expand: true })

      const first = matches![0]
      console.assert(first.tag === testInput.tag, 'mismatch', first, testInput)

      // console.log('input:', testInput.text)
      // console.log('topMatch', matches![0])
      // console.log('matches', matches)
    })
  },

  async runSuite() {
    const model = await TestRunner.train()
    await TestRunner.predict(model)
  }

}

TestRunner.runSuite()

export { TestRunner }
