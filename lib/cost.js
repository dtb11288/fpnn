const { ifElse, is, identity, add, multiply, map } = require('ramda')
const M = require('./matrix')

const costs = {
  squared: {
    forward: (targets, outputs) => M.subtract(targets, outputs).map(i => i * i / 2).reduce(add, 0),
    backward: (targets, outputs) => M.subtract(targets, outputs)
  },
  // don't use this, not sure it works
  // https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
  cross_entropy: {
    forward: (targets, outputs) => {
      const f = (t, o) => M.zipWith(multiply, t, o.map(Math.log))
      return -1 / targets.rows * M.add(
        f(targets, outputs),
        f(targets.map(i => 1 - i), outputs.map(i => 1 - i))
      ).reduce(add, 0)
    },
    backward: (targets, outputs) => {
      const inverse = mat => mat.map(i => 1 / i)
      return map(multiply(-1), M.add(
        M.zipWith(multiply, targets, inverse(outputs)),
        M.zipWith(multiply, targets.map(i => 1 - i), inverse(outputs.map(i => 1 - i)))
      ))
    }
  }
}

const get = ifElse(
  is(String),
  type => costs[type],
  identity
)

module.exports = {
  get
}
