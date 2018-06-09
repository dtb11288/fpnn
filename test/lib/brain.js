const B = require('../../lib/brain')
const path = require('path')
const fs = require('fs')
const saveFile = path.resolve('test/tmp/saved.json')
let savedBrain
try {
  savedBrain = require(saveFile)
} catch (e) {}

const learningRate = 0.01
const activation = 'leaky_relu'
const layersNodes = [2, 4, 1]
const babyBrain = savedBrain ? B.fromJSON(savedBrain) : B.initBrain(activation, layersNodes)

const correctData = [
  {
    inputs: [0, 0],
    targets: [0]
  },
  {
    inputs: [1, 1],
    targets: [0]
  },
  {
    inputs: [1, 0],
    targets: [1]
  },
  {
    inputs: [0, 1],
    targets: [1]
  }
]

let count = { 0: 0, 1: 0, 2: 0, 3: 0 }
const takeOne = data => {
  const i = Math.round(Math.random() * 1000) % 4
  count[i] += 1
  return data[i]
}

const data = (() => {
  let result = []
  for (let i = 0; i <= 30000; i++) {
    let e = takeOne(correctData)
    result.push(e)
  }
  return result
})()

const adultBrain = B.train(babyBrain, learningRate, data)
fs.writeFileSync(saveFile, JSON.stringify(B.toJSON(adultBrain)))

console.log(count)
console.log(adultBrain.predict([0, 0]), 'should be [0]')
console.log(adultBrain.predict([1, 1]), 'should be [0]')
console.log(adultBrain.predict([1, 0]), 'should be [1]')
console.log(adultBrain.predict([0, 1]), 'should be [1]')
