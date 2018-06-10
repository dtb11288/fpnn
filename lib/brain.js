const C = require('./cost')
const M = require('./matrix')
const L = require('./layer')
const {
  when, curry, pipe, aperture, reduce, map, compose, head,
  append, last, reverse, lt, length, defaultTo, __,
  prepend, addIndex, evolve, prop, converge, identity
} = require('ramda')
const reduceIndexed = addIndex(reduce)

const arrayToOneColMatrix = inputs => M.fromArray(inputs.length, 1, inputs)

const createBrain = (layers, info = {}) => {
  // inputs -> [outputs]
  const calculateOutputs = inputs => reduce((acc, { forward }) => {
    const inputs = last(acc).outputs
    const outputs = forward(inputs)
    return append(outputs, acc)
  }, [{ outputs: inputs }], layers)

  // inputs -> guesses
  const predict = compose(M.toArray, prop('outputs'), last, calculateOutputs, arrayToOneColMatrix)

  const { trained = 0 } = info
  const train = (config = {}, inputs, targets) => pipe(
    // forward
    arrayToOneColMatrix,
    calculateOutputs,

    // backward
    reverse,
    allOutputs => {
      const { outputs } = head(allOutputs)
      const { cost: costType, rate: learningRate } = config
      const costFuns = C.get(costType)
      const targetsMat = arrayToOneColMatrix(targets)
      const errors = costFuns.backward(targetsMat, outputs)
      const cost = costFuns.forward(targetsMat, outputs)
      const newLayers = pipe(
        reverse,
        reduceIndexed((acc, { backward, adjustLayer }, index) => {
          const errors = head(acc.errorsList)
          const isFirstLayer = layers.length - 1 === index
          const inputErrors = isFirstLayer ? undefined : backward(errors)
          const newLayer = adjustLayer(
            allOutputs[index + 1].outputs,
            allOutputs[index].preOutputs,
            allOutputs[index].outputs,
            errors,
            learningRate
          )
          return evolve({
            errorsList: isFirstLayer ? identity : prepend(inputErrors),
            layerList: prepend(newLayer)
          }, acc)
        }, { errorsList: [errors], layerList: [] }),
        prop('layerList')
      )(layers)
      return { cost, layers: newLayers }
    },

    // create brain with new layers
    ({ cost, layers }) => createBrain(layers, { cost, trained: trained + 1 })
  )(inputs)

  const toJSON = () => ({ info, layers: map(L.toJSON, layers) })

  return {
    info,
    toJSON,
    predict,
    train
  }
}

// so this will create stupid brain
const initBrain = (activation, nodes) => pipe(
  defaultTo([]),
  when(compose(lt(__, 2), length), () => { throw new Error('require input length at least 2') }),
  aperture(2),
  map(([input, output]) => L.initLayer(input, output, activation)),
  createBrain
)(nodes)

const trainMultiple = curry((brain, config, data) => reduce((brain, { inputs, targets }) => brain.train(config, inputs, targets), brain, data))

const fromJSON = converge(createBrain, [
  compose(map(L.fromJSON), prop('layers')),
  prop('info')
])
const toJSON = brain => brain.toJSON()

module.exports = {
  of: createBrain,
  fromJSON,
  toJSON,
  initBrain,
  train: trainMultiple
}
