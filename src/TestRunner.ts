import { TfClassifier, IMatch, ITaggedInput } from "./TfClassifier";
import { readCsvFile } from './FileUtils'
import chalk from 'chalk'
const debug = require('debug-levels')('TestRunner')
// const debug = require('debug-levels')('TestRunner')

// use a cached version of the models
// const useCache = false
const useCache = true

const TestRunner = {

  async prepare() {
    const testModel = new TfClassifier('testModel') // should be different in production
    await testModel.loadEncoder()
    await testModel.loadCsvInputs('./data/inputs/train.csv')
    await testModel.trainModel({ useCache: useCache })
    return testModel
  },

  async predict(testModel: TfClassifier) {
    const testLines = (await readCsvFile('./data/inputs/test.csv')).slice(0, 2)

    // console.log('passed\tactual\texpect\tconfidence\t\ttext')
    testLines.map(async testInput => {
      const matches: IMatch[] | undefined =
        await testModel.classify(testInput.text, { expand: true })

      const first = matches![0]
      console.assert(first.tag === testInput.tag, 'mismatch', first, testInput)
      // const passed = prediction?.tag === line.tag ? chalk.green('√ PASS') : chalk.red('X FAIL')
      // const output = (`${passed} \t${line.tag} \t${prediction?.tag} \t${prediction?.confidence}\t${line.text.trim()}`)
      // const compare = `${line.tag}\t${line.text}\n${prediction?.tag}\t${prediction?.found?.text}`
      // const trainSents = testModel.matchingInputs(prediction!.tag)

      console.log('input:', testInput.text)
      console.log('topMatch', matches![0])
      console.log('matches', matches)
    })
  },

  async runSuite() {
    const testModel = await TestRunner.prepare()
    await TestRunner.predict(testModel)
  }

}

TestRunner.runSuite()

export { TestRunner }
