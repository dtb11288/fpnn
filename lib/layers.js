const A = require('./activation')
const M = require('./matrix')
const { compose, map, pipe, multiply } = require('ramda')

const createLayer = (weights, biases, activationType) => {
  const { activation, deactivation } = A.get(activationType)
  // inputs -> outputs
  const forward = compose(map(activation), M.add(biases), M.multiply(weights))

  // outputErrors -> inputErrors
  const backward = M.multiply(M.transpose(weights))

  // inputs -> outputs -> errors -> learningRate -> layer
  const adjustLayer = (inputs, outputs, errors, learningRate) => {
    const biasesDelta = pipe(
      map(multiply(learningRate)),
      M.zipWith(multiply, outputs.map(deactivation))
    )(errors)
    const weightsDelta = M.multiply(biasesDelta, (M.transpose(inputs)))
    return createLayer(M.add(weights, weightsDelta), M.add(biases, biasesDelta), activationType)
  }

  const print = notice => weights.print(notice)
  const toJSON = () => ({ weights: weights.toJSON(), biases: biases.toJSON(), activation: A.toJSON(activationType) })

  return { forward, backward, adjustLayer, print, toJSON }
}

const initLayer = (input, output, config) => createLayer(M.random(output, input), M.random(output, 1), config)

const toJSON = layer => layer.toJSON()
const fromJSON = data => createLayer(
  M.fromJSON(data.weights),
  M.fromJSON(data.biases),
  A.fromJSON(data.activation)
)

module.exports = {
  toJSON,
  fromJSON,
  createLayer,
  initLayer
}
