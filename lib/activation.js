const { ifElse, is, identity, map } = require('ramda')

const activations = {
  tanh: {
    activation: x => Math.tanh(x),
    deactivation: y => 1 - (y * y),
    fx: true
  },
  sigmoid: {
    activation: x => 1 / (1 + Math.exp(-x)),
    deactivation: y => y * (1 - y),
    fx: true
  },
  relu: {
    activation: x => x > 0 ? x : 0,
    deactivation: x => x > 0 ? 1 : 0,
    fx: false
  },
  leaky_relu: {
    activation: x => x > 0 ? x : x * 0.01,
    deactivation: x => x > 0 ? 1 : 0.01,
    fx: false
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
