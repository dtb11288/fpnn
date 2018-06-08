const { ifElse, is, identity, map } = require('ramda')
const sigmoid = x => 1 / (1 + Math.exp(-x))
const desigmoid = x => {
  // const y = sigmoid(x)
  const y = x
  return y * (1 - y)
}

const activations = {
  sigmoid: {
    activation: sigmoid,
    deactivation: desigmoid
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
