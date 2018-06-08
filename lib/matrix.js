const {
  curry, add, zipWith, any, chain, identity, pipe, map, range,
  splitEvery, multiply, transpose, compose, prop, sum, subtract
} = require('ramda')

const createMatrix = array2D => {
  const rows = array2D.length
  if (!rows > 0 || !array2D[0]) {
    throw new Error('invalid matrix')
  }
  const cols = array2D[0].length
  if (any(cv => cv.length !== cols, array2D)) {
    throw new Error('invalid matrix')
  }
  const toArray = () => chain(identity, array2D)
  const mapMatrix = fn => createMatrix(map(map(fn), array2D))
  const print = (notice = 'matrix') => console.log(`${notice}`, array2D)
  const toJSON = () => ({ rows, cols, data: array2D })
  return {
    toJSON,
    'fantasy-land/map': mapMatrix,
    rows,
    cols,
    data: array2D,
    toArray,
    map: mapMatrix,
    print
  }
}

const fromArray = curry((rows, cols, data) => {
  if ((data.length) !== (rows * cols)) {
    throw new Error('invalid matrix')
  }
  return pipe(
    splitEvery(cols),
    createMatrix
  )(data)
})

const randomMatrix = (rows, cols) => pipe(
  range(0),
  map(() => Math.random() * 2 - 1), // -1 ... 1
  data => fromArray(rows, cols, data)
)(rows * cols)

const zipMatrix = curry((fn, mat1, mat2) => {
  if (mat1.rows !== mat2.rows || mat1.cols !== mat2.cols) {
    throw new Error('size mismatch')
  }
  return createMatrix(zipWith(zipWith(fn), mat1.data, mat2.data))
})

const addMatrix = zipMatrix(add)

const subtractMatrix = zipMatrix(subtract)

const combineRow = compose(sum, zipWith(multiply))

const multiplyMatrix = curry((mat1, mat2) => {
  if (mat1.cols !== mat2.rows) {
    throw new Error('size mismatch for multiplying')
  }
  const mat2T = transposeMatrix(mat2)
  const data = map(ar => map(bc => combineRow(ar, bc), mat2T.data), mat1.data)
  return createMatrix(data)
})

const transposeMatrix = compose(createMatrix, transpose, prop('data'))

const fromJSON = compose(createMatrix, prop('data'))
const toJSON = mat => mat.toJSON()

module.exports = {
  toJSON,
  fromJSON,
  of: createMatrix,
  add: addMatrix,
  subtract: subtractMatrix,
  multiply: multiplyMatrix,
  transpose: transposeMatrix,
  random: randomMatrix,
  fromArray,
  zipWith: zipMatrix,
  toArray: mat => mat.toArray()
}
