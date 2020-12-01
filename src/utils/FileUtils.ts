const csv = require('csv-parser');
const fs = require('fs');
import * as path from 'path'

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
  if (fs.existsSync(dirPath)) { return }
  // else
  fs.mkdirSync(dirPath, { recursive: true })
}

export { readCsvFile, ensureDirectory }
