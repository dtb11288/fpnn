const M = require('./matrix')
const L = require('./layers')
const {
  when, curry, pipe, aperture, reduce, map, compose, head,
  append, last, reverse, lt, length, defaultTo, __,
  prepend, addIndex, init, prop
} = require('ramda')
const mapIndexed = addIndex(map)

const arrayToOneColMatrix = inputs => M.fromArray(inputs.length, 1, inputs)

const createBrain = layers => {
  // inputs -> [outputs]
  const calculateOutputs = inputs => reduce((acc, { forward }) => {
    const inputs = last(acc)
    const outputs = forward(inputs)
    return append(outputs, acc)
  }, [inputs], layers)

  // inputs -> guesses
  const predict = compose(M.toArray, last, calculateOutputs, arrayToOneColMatrix)

  const train = (learningRate, inputs, targets) => pipe(
    arrayToOneColMatrix,
    calculateOutputs,
    allOutputs => {
      const outputs = last(allOutputs)
      const errors = calculateErrors(arrayToOneColMatrix(targets), outputs)
      const allErrors = pipe(
        reverse,
        init, // don't calculate error for the first layer
        reduce((acc, { backward }) => {
          const errors = head(acc)
          const inputErrors = backward(errors)
          return prepend(inputErrors, acc)
        }, [errors])
      )(layers)
      return mapIndexed((layer, index) => layer.adjustLayer(allOutputs[index], allOutputs[index + 1], allErrors[index], learningRate), layers)
    },
    createBrain
  )(inputs)

  const toJSON = () => ({ layers: map(L.toJSON, layers) })

  return {
    toJSON,
    predict,
    train
  }
}

const calculateErrors = curry((targets, outputs) => M.subtract(targets, outputs))

// so this will create stupid brain
const initBrain = (activation, nodes) => pipe(
  defaultTo([]),
  when(compose(lt(__, 2), length), () => { throw new Error('require input length at least 2') }),
  aperture(2),
  map(([input, output]) => L.initLayer(input, output, activation)),
  createBrain
)(nodes)

const trainMultiple = curry((brain, learningRate, data) => reduce((brain, { inputs, targets }) => brain.train(learningRate, inputs, targets), brain, data))

const fromJSON = compose(createBrain, map(L.fromJSON), prop('layers'))
const toJSON = brain => brain.toJSON()

module.exports = {
  fromJSON,
  toJSON,
  initBrain,
  train: trainMultiple
}
