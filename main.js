class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

class NeuralNetwork {
  constructor() {
    this.layers = []
    this.learningRate = 0.01
    this.sizeInput = 0
    this.sizeOutput = 1
    this.activationFunction = new ActivationFunction( // Default Sigmoid
      x => 1 / (1 + Math.exp(- x)),
      y => y * (1 - y)
    )
  }

  defineSizeInput = size => this.sizeInput = size
  defineSizeOutput = size => this.sizeOutput = size
  defineActivationFunction = activationFunction => this.activationFunction = activationFunction

  add = layer => this.layers.push(layer)

  forward = inputs => { // TODO: faire avec reduce ?
    let outputs = new Matrix(inputs.length, 1)
    outputs.setFromArray(inputs)
    this.layers.forEach(layer => {
      let result = layer.dot(outputs)
      // result.add(bias)
      result.apply(this.activationFunction.func)
      console.log(result)
      outputs = result
    })
    // return outputs.toArray()
    return outputs
  }

  train = (inputArray, targetArray) => {
    let prediction = this.forward(inputArray)
    let target = new Matrix(targetArray.length, 1)
    target.setFromArray(targetArray)
    let errors = target.subtract(prediction)
    console.table(prediction.values)
    let gradients = Matrix.map(prediction, this.activationFunction.dfunc)
    console.table(prediction.values)

    console.table(gradients.values)
    gradients.multiply(errors)
    gradients.multiply(this.learningRate)
    console.log(gradients)
  }
}

const train = (inputsArray, targetArray) => {
  let outputs = Matrix.toMatrix(inputsArray)
  let listResult = []
  listResult.push(outputs)
  this.layers.forEach( (layer, index) => {
    let result = Matrix.multiply(layer, outputs)
    result.add(this.listBias[index])
    result.applyActivationFunction(this.activationFunction.func)
    listResult.push(result)
    outputs = result
  })
  let targets = Matrix.toMatrix(targetArray)

  let errors = Matrix.subtract(targets, outputs)
  let gradients = Matrix.map(outputs, this.activationFunction.dfunc)
  gradients.multiply(errors)
  gradients.multiply(this.learning_rate)

  let hiddenTransposed = Matrix.transpose(listResult[this.layers.length-1])
  let weightsDeltas = Matrix.multiply(gradients, hiddenTransposed)
  this.listBias[this.layers.length-1].add(gradients)
  this.layers[this.layers.length-1].add(weightsDeltas)

  for( let i = this.layers.length-1; i > 0; i--){ // i stop à 1
    let whoT = Matrix.transpose(this.layers[i]);
    let err = Matrix.multiply(whoT, errors)
    errors = err

    let gradients = Matrix.map(listResult[i], this.activationFunction.dfunc)
    gradients.multiply(errors)
    gradients.multiply(this.learning_rate)

    let hiddenTransposed = Matrix.transpose(listResult[i-1])
    let weightsDeltas = Matrix.multiply(gradients, hiddenTransposed)

    this.layers[i-1].add(weightsDeltas)
    this.listBias[i-1].add(gradients)
  }
}



// TODO: utiliser Uint8array
class Matrix {
  constructor(nbRows, nbCols, values) {
    this.nbRows = nbRows
    this.nbCols = nbCols
    this.values = new Array(this.nbRows).fill().map(elem => new Array(this.nbCols).fill().map(value => - 1 + 2 * Math.random()))
    this.name = ''
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

  apply = func => this.values = this.values.map(row => row.map(value => func(value)))

  setFromArray = array => this.values.forEach((row, index) => row[0] = array[index])
  toArray = () => this.values.flat()

  subtract = matrix => {
    let result = new Matrix(this.nbRows, matrix.nbCols);
    for (let i = 0; i < result.nbRows; i++) {
      for (let j = 0; j < result.nbCols; j++) {
        result.values[i][j] = this.values[i][j] - matrix.values[i][j];
      }
    }
    return result
  }

  multiply = n => {
    if (n instanceof Matrix) {
      // hadamard product
      for (let i = 0; i < this.nbRows; i++) {
        for (let j = 0; j < this.nbCols; j++) {
          this.values[i][j] *= n.values[i][j]
        }
      }
    }
    else {
      // Scalar product
      for (let i = 0; i < this.nbRows; i++) {
        for (let j = 0; j < this.nbCols; j++) {
          this.values[i][j] *= n
        }
      }
    }
  }

  static map(matrix, func) {
    let result = new Matrix(matrix.nbRows, matrix.nbCols)
    for (let i = 0; i < matrix.nbRows; i++) {
      for (let j = 0; j < matrix.nbCols; j++) {
        result.values[i][j] = func(matrix.values[i][j])
      }
    }
    return result
  }
}

// TODO: define size input output automatic
let m1 = new Matrix(2, 2)
let m2 = new Matrix(2, 2)
m1.values = [[1, 2], [4, 5]]
m2.values = [[7, 8], [9, 10]]
let model = new NeuralNetwork()
model.defineSizeInput(2)
model.add(m1)
model.add(m2)
console.log(model)

model.train([4, 6], [0.5, 0.8])

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
