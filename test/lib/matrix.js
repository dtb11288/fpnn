const M = require('../../lib/matrix')
const mat1 = M.random(2, 3).map(i => Math.floor(i * 9))
const mat2 = M.random(3, 1).map(i => Math.floor(i * 9))

mat1.print()
mat2.print()
// M.add(mat1, mat1).print()
// M.transpose(mat2).print()
M.multiply(mat1, mat2).print()
