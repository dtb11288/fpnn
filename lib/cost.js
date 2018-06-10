const { ifElse, is, identity, add } = require('ramda')
const M = require('./matrix')

const costs = {
  squared: {
    forward: (targets, outputs) => M.subtract(targets, outputs).map(i => i * i / 2).reduce(add, 0),
    backward: (targets, outputs) => M.subtract(targets, outputs)
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
