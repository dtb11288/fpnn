const M = require('./matrix')
const L = require('./layers')
const {
  when, curry, pipe, aperture, reduce, map, compose, head,
  append, last, reverse, lt, length, defaultTo, __,
  prepend, addIndex, init, prop, add
} = require('ramda')
const mapIndexed = addIndex(map)

const arrayToOneColMatrix = inputs => M.fromArray(inputs.length, 1, inputs)

const createBrain = (layers, lastCost) => {
  // inputs -> [outputs]
  const calculateOutputs = inputs => reduce((acc, { forward }) => {
    const inputs = last(acc).outputs
    const outputs = forward(inputs)
    return append(outputs, acc)
  }, [{ outputs: inputs }], layers)

  // inputs -> guesses
  const predict = compose(M.toArray, prop('outputs'), last, calculateOutputs, arrayToOneColMatrix)

  const train = (learningRate, inputs, targets) => pipe(
    arrayToOneColMatrix,
    calculateOutputs,
    allOutputs => {
      const { outputs } = last(allOutputs)
      const errors = calculateErrors(arrayToOneColMatrix(targets), outputs)
      const cost = errors.map(i => i * i / 2).reduce(add, 0)
      const allErrors = pipe(
        reverse,
        init, // don't calculate error for the first layer
        reduce((acc, { backward }) => {
          const errors = head(acc)
          const inputErrors = backward(errors)
          return prepend(inputErrors, acc)
        }, [errors])
      )(layers)
      const newLayers = mapIndexed((layer, index) => layer.adjustLayer(
        allOutputs[index].outputs,
        allOutputs[index + 1].preOutputs,
        allOutputs[index + 1].outputs,
        allErrors[index],
        learningRate
      ), layers)
      return {
        cost,
        layers: newLayers
      }
    },
    ({ cost, layers }) => createBrain(layers, cost)
  )(inputs)

  const toJSON = () => ({ layers: map(L.toJSON, layers) })

  return {
    lastCost,
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
