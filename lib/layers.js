const M = require('./matrix')
const { compose, map, pipe, multiply } = require('ramda')
const sigmoid = i => 1 / (1 + Math.exp(-i))
const desigmoid = i => i * (1 - i)

const createLayer = (weights, biases) => {
  // inputs -> outputs
  const forward = compose(map(sigmoid), M.add(biases), M.multiply(weights))

  // outputErrors -> inputErrors
  const backward = M.multiply(M.transpose(weights))

  // inputs -> outputs -> errors -> learningRate -> layer
  const adjustLayer = (inputs, outputs, errors, learningRate) => {
    const biasesDelta = pipe(
      map(multiply(learningRate)),
      M.zipWith(multiply, outputs.map(desigmoid))
    )(errors)
    const weightsDelta = M.multiply(biasesDelta, (M.transpose(inputs)))
    return createLayer(M.add(weights, weightsDelta), M.add(biases, biasesDelta))
  }

  const print = notice => weights.print(notice)
  const toJSON = () => ({ weights: weights.toJSON(), biases: biases.toJSON() })

  return { forward, backward, adjustLayer, print, toJSON }
}

const initLayer = (input, output) => createLayer(M.random(output, input), M.random(output, 1))
const toJSON = layer => layer.toJSON()
const fromJSON = data => createLayer(M.fromJSON(data.weights), M.fromJSON(data.biases))

module.exports = {
  toJSON,
  fromJSON,
  createLayer,
  initLayer
}
