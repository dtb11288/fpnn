const { ifElse, is, identity, map } = require('ramda')
const M = require('./matrix')

const activations = {
  tanh: {
    forward: mat => mat.map(Math.tanh),
    backward: mat => mat.map(y => 1 - (y * y)),
    fx: true
  },
  sigmoid: {
    forward: mat => mat.map(x => 1 / (1 + Math.exp(-x))),
    backward: mat => mat.map(y => y * (1 - y)),
    fx: true
  },
  relu: {
    forward: mat => mat.map(x => x > 0 ? x : 0),
    backward: mat => mat.map(x => x > 0 ? 1 : 0)
  },
  leaky_relu: {
    forward: mat => mat.map(x => x > 0 ? x : x * 0.01),
    backward: mat => mat.map(x => x > 0 ? 1 : 0.01)
  },
  softmax: {
    forward: mat => {
      const array = mat.toArray().map(Math.exp)
      const sum = array.reduce((a, b) => a + b)
      return M.fromArray(mat.rows, mat.cols, array.map(i => i / sum))
    },
    backward: mat => {
      const array = mat.toArray().map(Math.exp)
      const sum = array => array.reduce((a, b) => a + b)
      const sumArray = sum(array)
      return M.fromArray(mat.rows, mat.cols, array.map((i, index) => {
        const otherSum = sum(array.slice().splice(index, 1))
        return i * otherSum / sumArray
      }))
    }
  }
}

const toJSON = ifElse(
  is(String),
  identity,
  map(fn => fn.toString())
)

const fromJSON = ifElse(
  is(String),
  identity,
  map(fnStr => (new Function(`return ${fnStr}`))())
)

const get = ifElse(
  is(String),
  type => activations[type],
  identity
)

module.exports = {
  get,
  fromJSON,
  toJSON
}
