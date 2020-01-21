class NeuralNetwork {
  constructor() {
    this.layers = []
    this.sizeInput = 0
    this.sizeOutput = 1
    this.activationFunction = 'sigmoid'
  }

  defineSizeInput = size => this.sizeInput = size
  defineSizeOutput = size => this.sizeOutput = size
  defineActivationFunction = nameActivationFunction => {
    this.activationFunction = nameActivationFunction
  }
  
  add = layer => this.layers.push(layer)

  forward = inputs => { // TODO: faire avec reduce ?
    let outputs = new Matrix(inputs.length, 1)
    outputs.setFromArray(inputs)
    this.layers.forEach(layer => {
      console.table(outputs.values)
      let result = layer.dot(outputs)
      outputs = result
    })
    return outputs.toArray()
  }
}


// let outputs = Matrix.toMatrix(inputs)
// this.layers.forEach( (layer, index) => {
//   let result = layer.dot(outputs)//(layer, outputs)
//   // result.add(this.listBias[index])
//   // result.applyActivationFunction(this.activationFunction.func)
//   outputs = result
// })
// return Matrix.toArray(outputs)

// class ActivationFunction {
//   constructor(func, dfunc) {
//     this.func = func;
//     this.dfunc = dfunc;
//   }
// }
//
// const sigmoid = new ActivationFunction(
//   x => 1 / (1 + Math.exp(-x)),
//   y => y * (1 - y)
// )


class Matrix {
  constructor(nbRows, nbCols, values) {
    this.nbRows = nbRows
    this.nbCols = nbCols
    this.values = new Array(this.nbRows).fill().map(elem => new Array(this.nbCols).fill().map(value => - 1 + 2 * Math.random()))
    this.name = ''
    console.log(this.values)
  }

  dot = matrix => {
    let result = new Matrix(this.nbRows, matrix.nbCols)
    for (let i = 0; i < this.nbRows; i++) {
      for (let j = 0; j < matrix.nbCols; j++) {
        let sum = 0
        for (let k = 0; k < this.nbCols; k++) {
          sum += this.values[i][k] * matrix.values[k][j]
        }
        result.values[i][j] = sum
      }
    }
    return result
  }

  setFromArray = values => this.values.forEach(row => row[0] = 1)
  toArray = () => this.values.flat()
}

//   static multiply(m1, m2){
//     let matrix = new Matrix(m1.rows, m2.cols)
//     if(m1.cols != m2.rows){
//       console.error('Cannot multiply m1 and m2')
//       return
//     }
//     for(let i=0; i<m1.rows; i++){
//       for(let j=0; j<m2.cols; j++){
//         let sum = 0
//         for(let k=0; k<m1.cols; k++){
//           sum += m1.data[i][k] * m2.data[k][j]
//         }
//         matrix.data[i][j] = sum
//       }
//     }
//     return matrix
//   }

let m1 = new Matrix(2, 2)
m1.values = [[1, 2], [4, 5]]
let m2 = new Matrix(2, 2)
m2.values = [[7, 8], [9, 10]]
// console.log(m1.dot(m2))
let model = new NeuralNetwork()
model.defineSizeInput(2)
// model.addLayer()
model.add(m1)
// model.add(m2)
console.log(model.forward([1, 1]))

// class Matrix {
//   constructor(rows, cols){
//     this.rows = rows
//     this.cols = cols
//     this.data = []
//     for(let i=0; i<rows; i++){
//       this.data[i] = []
//       for(let j=0; j<cols; j++){
//         this.data[i][j] = -1 + Math.random()*2
//       }
//     }
//   }
//
//   add(n) {
//     if (n instanceof Matrix) {
//       for (let i = 0; i < this.rows; i++) {
//         for (let j = 0; j < this.cols; j++) {
//           this.data[i][j] += n.data[i][j];
//         }
//       }
//     } else {
//       for (let i = 0; i < this.rows; i++) {
//         for (let j = 0; j < this.cols; j++) {
//           this.data[i][j] += n;
//         }
//       }
//     }
//   }
//
//   static subtract(a, b) {
//     // Return a new Matrix a-b
//     let result = new Matrix(a.rows, a.cols);
//     for (let i = 0; i < result.rows; i++) {
//       for (let j = 0; j < result.cols; j++) {
//         result.data[i][j] = a.data[i][j] - b.data[i][j];
//       }
//     }
//     return result;
//   }
//
//   applyActivationFunction(func){
//     for (let i = 0; i < this.rows; i++) {
//       for (let j = 0; j < this.cols; j++) {
//         let val = this.data[i][j];
//         this.data[i][j] = func(val);
//       }
//     }
//   }
//   static map(matrix, func) {
//     let result = new Matrix(matrix.rows, matrix.cols);
//     // Apply a function to every element of matrix
//     for (let i = 0; i < matrix.rows; i++) {
//       for (let j = 0; j < matrix.cols; j++) {
//         let val = matrix.data[i][j];
//         result.data[i][j] = func(val);
//       }
//     }
//     return result;
//   }
//
//   static multiply(m1, m2){
//     let matrix = new Matrix(m1.rows, m2.cols)
//     if(m1.cols != m2.rows){
//       console.error('Cannot multiply m1 and m2')
//       return
//     }
//     for(let i=0; i<m1.rows; i++){
//       for(let j=0; j<m2.cols; j++){
//         let sum = 0
//         for(let k=0; k<m1.cols; k++){
//           sum += m1.data[i][k] * m2.data[k][j]
//         }
//         matrix.data[i][j] = sum
//       }
//     }
//     return matrix
//   }
//
//   multiply(n) {
//     if (n instanceof Matrix) {
//       // hadamard product
//       for (let i = 0; i < this.rows; i++) {
//         for (let j = 0; j < this.cols; j++) {
//           this.data[i][j] *= n.data[i][j];
//         }
//       }
//     }
//     else {
//       // Scalar product
//       for (let i = 0; i < this.rows; i++) {
//         for (let j = 0; j < this.cols; j++) {
//           this.data[i][j] *= n;
//         }
//       }
//     }
//   }
//
//   static transpose(matrix) {
//     let result = new Matrix(matrix.cols, matrix.rows);
//     for (let i = 0; i < matrix.rows; i++) {
//       for (let j = 0; j < matrix.cols; j++) {
//         result.data[j][i] = matrix.data[i][j];
//       }
//     }
//     return result;
//   }
//
//   static toMatrix(array){
//     let matrix = new Matrix(array.length, 1)
//     for(let i=0; i<array.length; i++){
//       matrix.data[i][0] = array[i]
//     }
//     return matrix
//   }
//
//   static toArray(matrix){
//     if(matrix.cols != 1){
//       console.error("Cannot transform multi dimensionnal matrix into array")
//       return void 0
//     }
//     let array = []
//     for(let i=0; i<matrix.rows; i++){
//       array.push(matrix.data[i][0])
//     }
//     return array
//   }
//
//   copy(){
//     let m = new Matrix(this.rows, this.cols);
//     for (let i=0; i<this.rows;i++) {
//       for (let j=0; j<this.cols;j++) {
//         m.data[i][j] = this.data[i][j];
//       }
//     }
//     return m;
//   }
//
//   mutate(mutationRate){
//     for (let i=0; i<this.rows; i++) {
//       for (let j=0; j<this.cols; j++) {
//         let rdm = Math.random()
//         if(rdm < mutationRate){
//           this.data[i][j] = -1 + Math.random()*2
//         }
//       }
//     }
//   }
// }
