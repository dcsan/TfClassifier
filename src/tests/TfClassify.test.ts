import { readCsvFile } from '../FileUtils'
import { TfClassifier, IMatch, ITaggedInput } from "../TfClassifier";

const debug = require('debug-levels')('TfCl.test')

const useCache = true
// let testModel: TfClassifier

const prepareModel = async () => {
  const model = new TfClassifier('testModel') // should be different in production
  await model.loadEncoder()
  debug.log('loaded test')
  return model
}

// const trainJson = async (model: TfClassifier) => {
//   const trainData = await readCsvFile('./data/inputs/test.csv')
//   await model.trainModel({
//     // json: trainData,
//     useCache
//   })
// }


const checkMatches = async (model) => {

  const testLines: any[] = await readCsvFile('../data/inputs/test.csv', __dirname)
  // console.log('passed\tactual\texpect\tconfidence\t\ttext')
  testLines.map(async testInput => {
    const text = testInput.text
    debug.log('testing', text)
    const matches: IMatch[] | undefined = await model.classify(text, { expand: true })

    const first = matches![0]
    console.assert(first.tag === testInput.tag, 'mismatch', first, testInput)

    console.log('input:', testInput.text)
    console.log('topMatch', matches![0])
    console.log('matches', matches)
  })
}


test('train on csv', async () => {
  const model = await prepareModel()
  await model.loadCsvInputs('./data/inputs/train.csv')
  await model.trainModel({ useCache: false })
  await checkMatches(model)
})

// xtest('train json and match', async () => {
//   const model = await prepareModel()
//   await trainJson(model)
//   await checkMatches(model)
// })


