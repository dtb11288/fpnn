const A = require('./activation')
const M = require('./matrix')
const { compose, map, pipe, multiply } = require('ramda')

const createLayer = (weights, biases, activationType) => {
  const activation = A.get(activationType)

  // inputs -> { preOutputs, outputs }
  const forward = inputs => {
    // return both preOutputs and outputs to avoid calculate multiple times
    const preOutputs = compose(M.add(biases), M.multiply(weights))(inputs)
    const outputs = activation.forward(preOutputs)
    return { preOutputs, outputs }
  }

  // outputErrors -> inputErrors
  const backward = M.multiply(M.transpose(weights))

  // inputs -> preOutputs -> outputs -> errors -> learningRate -> layer
  const adjustLayer = (inputs, preOutputs, outputs, errors, learningRate) => {
    const gradients = pipe(
      activation.backward,
      M.zipWith(multiply, errors),
      map(multiply(learningRate))
    )(activation.fx ? outputs : preOutputs)
    const biasesDelta = gradients
    const weightsDelta = M.multiply(gradients, (M.transpose(inputs)))
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
