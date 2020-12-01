import { readCsvFile } from '../utils/FileUtils'

const debug = require('debug-levels')('TfCl.test')

test('loading csv', async () => {
  let blob: any[] = await readCsvFile('../data/inputs/test.csv', __dirname)
  debug.log('blob', blob)
  expect(blob[0].tag).toBe('READ')
  expect(blob[0].text).toBe('I need to read more')
})

