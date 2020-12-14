const csv = require('csv-parser');
const fs = require('fs');
import * as path from 'path'
const debug = require('debug-levels')('FileUtils')

const readCsvFile = (fp: string, basePath: string = __dirname): Promise<any[]> => {
  const fullPath = path.join(basePath, fp)

  return new Promise((resolve, reject) => {
    let lines: string[] = []
    const opts = {
      mapValues: ({ header, index, value }) => value.trim()
    }
    fs.createReadStream(fullPath)
      .pipe(csv(opts))
      .on('data', (row) => {
        lines.push(row)
      })
      .on('end', () => {
        resolve(lines)
      });
  })

}

const ensureDirectory = (dirPath) => {
  if (fs.existsSync(dirPath)) {
    debug.info('ensure dir OK', dirPath)
    return
  }
  // else
  debug.log('creating dir', dirPath)
  try {

    fs.mkdirSync(dirPath, { recursive: true })
    if (!fs.existsSync(dirPath)) {
      debug.error('FAIL to create dir', dirPath)
      return false
    }
    return true
  } catch (err) {
    debug.error('ERROR on creating dir', dirPath)
    debug.error(err)
    return false
  }
}

export { readCsvFile, ensureDirectory }
