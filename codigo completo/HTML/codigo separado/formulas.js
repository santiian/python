// Configuración inicial de MathJax
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    options: {
        processHtmlClass: 'tex2jax_process',
        processEscapes: true
    }
};

// Clase Matrix mejorada
class Matrix {
    constructor(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Los datos de la matriz deben ser un array no vacío');
        }
        this.data = data.map(row => [...row]); // Deep copy
        this.rows = data.length;
        this.cols = data[0].length;

        // Validar que todas las filas tengan la misma longitud
        if (!data.every(row => row.length === this.cols)) {
            throw new Error('Todas las filas deben tener la misma cantidad de columnas');
        }
    }

    static add(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Las matrices deben tener las mismas dimensiones para sumar');
        }
        const result = [];
        for (let i = 0; i < a.rows; i++) {
            result[i] = [];
            for (let j = 0; j < a.cols; j++) {
                result[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return new Matrix(result);
    }

    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Las matrices deben tener las mismas dimensiones para restar');
        }
        const result = [];
        for (let i = 0; i < a.rows; i++) {
            result[i] = [];
            for (let j = 0; j < a.cols; j++) {
                result[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return new Matrix(result);
    }

    static multiply(a, b) {
        if (a.cols !== b.rows) {
            throw new Error('El número de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz');
        }
        const result = [];
        for (let i = 0; i < a.rows; i++) {
            result[i] = [];
            for (let j = 0; j < b.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }

    transpose() {
        const result = [];
        for (let j = 0; j < this.cols; j++) {
            result[j] = [];
            for (let i = 0; i < this.rows; i++) {
                result[j][i] = this.data[i][j];
            }
        }
        return new Matrix(result);
    }

    determinant() {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para calcular el determinante');
        }
        if (this.rows === 1) {
            return this.data[0][0];
        }
        if (this.rows === 2) {
            return this.data[0][0] * this.data[1][1] - this.data[0][1] * this.data[1][0];
        }
        let det = 0;
        for (let j = 0; j < this.cols; j++) {
            const submatrix = [];
            for (let i = 1; i < this.rows; i++) {
                submatrix[i - 1] = [];
                for (let k = 0; k < this.cols; k++) {
                    if (k !== j) {
                        submatrix[i - 1].push(this.data[i][k]);
                    }
                }
            }
            det += Math.pow(-1, j) * this.data[0][j] * new Matrix(submatrix).determinant();
        }
        return det;
    }

    scalarMultiply(scalar) {
        const result = [];
        for (let i = 0; i < this.rows; i++) {
            result[i] = [];
            for (let j = 0; j < this.cols; j++) {
                result[i][j] = this.data[i][j] * scalar;
            }
        }
        return new Matrix(result);
    }

    power(n) {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para calcular la potencia');
        }
        if (n === 0) {
            const identity = [];
            for (let i = 0; i < this.rows; i++) {
                identity[i] = [];
                for (let j = 0; j < this.cols; j++) {
                    identity[i][j] = i === j ? 1 : 0;
                }
            }
            return new Matrix(identity);
        }
        let result = new Matrix(this.data);
        for (let i = 1; i < n; i++) {
            result = Matrix.multiply(result, this);
        }
        return result;
    }

    inverse() {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para calcular la inversa');
        }
        const det = this.determinant();
        if (det === 0) {
            throw new Error('La matriz no es invertible (determinante es cero)');
        }
        const adjugate = this.adjugate();
        const result = [];
        for (let i = 0; i < this.rows; i++) {
            result[i] = [];
            for (let j = 0; j < this.cols; j++) {
                result[i][j] = adjugate.data[i][j] / det;
            }
        }
        return new Matrix(result);
    }

    adjugate() {
        const cofactorMatrix = this.cofactorMatrix();
        return cofactorMatrix.transpose();
    }

    cofactorMatrix() {
        const result = [];
        for (let i = 0; i < this.rows; i++) {
            result[i] = [];
            for (let j = 0; j < this.cols; j++) {
                const submatrix = [];
                for (let m = 0; m < this.rows; m++) {
                    if (m !== i) {
                        submatrix.push([]);
                        for (let n = 0; n < this.cols; n++) {
                            if (n !== j) {
                                submatrix[submatrix.length - 1].push(this.data[m][n]);
                            }
                        }
                    }
                }
                result[i][j] = Math.pow(-1, i + j) * new Matrix(submatrix).determinant();
            }
        }
        return new Matrix(result);
    }

    rank() {
        const rref = this.rref();
        let rank = 0;
        for (let i = 0; i < rref.rows; i++) {
            let allZeros = true;
            for (let j = 0; j < rref.cols; j++) {
                if (rref.data[i][j] !== 0) {
                    allZeros = false;
                    break;
                }
            }
            if (!allZeros) {
                rank++;
            }
        }
        return rank;
    }

    rref() {
        const matrix = new Matrix(this.data);
        let lead = 0;
        for (let r = 0; r < matrix.rows; r++) {
            if (lead >= matrix.cols) {
                return matrix;
            }
            let i = r;
            while (matrix.data[i][lead] === 0) {
                i++;
                if (i === matrix.rows) {
                    i = r;
                    lead++;
                    if (lead === matrix.cols) {
                        return matrix;
                    }
                }
            }
            [matrix.data[i], matrix.data[r]] = [matrix.data[r], matrix.data[i]];
            const lv = matrix.data[r][lead];
            for (let j = 0; j < matrix.cols; j++) {
                matrix.data[r][j] /= lv;
            }
            for (let i = 0; i < matrix.rows; i++) {
                if (i !== r) {
                    const lv = matrix.data[i][lead];
                    for (let j = 0; j < matrix.cols; j++) {
                        matrix.data[i][j] -= lv * matrix.data[r][j];
                    }
                }
            }
            lead++;
        }
        return matrix;
    }

    trace() {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para calcular la traza');
        }
        let trace = 0;
        for (let i = 0; i < this.rows; i++) {
            trace += this.data[i][i];
        }
        return trace;
    }

    eigenvalues() {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para calcular los eigenvalores');
        }
        if (this.rows === 1) {
            return [this.data[0][0]];
        }
        if (this.rows === 2) {
            const a = this.data[0][0];
            const b = this.data[0][1];
            const c = this.data[1][0];
            const d = this.data[1][1];
            const trace = a + d;
            const det = a * d - b * c;
            const discriminant = trace * trace - 4 * det;
            if (discriminant >= 0) {
                return [(trace + Math.sqrt(discriminant)) / 2, (trace - Math.sqrt(discriminant)) / 2];
            } else {
                return [(trace + Math.sqrt(-discriminant)) / 2 + 'i', (trace - Math.sqrt(-discriminant)) / 2 + 'i'];
            }
        }
        // Para matrices más grandes, se puede usar el método de la potencia o QR
        throw new Error('Cálculo de eigenvalores no implementado para matrices mayores a 2x2');
    }

    luDecomposition() {
        if (this.rows !== this.cols) {
            throw new Error('La matriz debe ser cuadrada para la descomposición LU');
        }
        const n = this.rows;
        const L = new Array(n).fill().map(() => new Array(n).fill(0));
        const U = new Array(n).fill().map(() => new Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            // Descomposición de Doolittle
            // U
            for (let j = i; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = this.data[i][j] - sum;
            }

            // L
            for (let j = i; j < n; j++) {
                if (i === j) {
                    L[i][i] = 1;
                } else {
                    let sum = 0;
                    for (let k = 0; k < i; k++) {
                        sum += L[j][k] * U[k][i];
                    }
                    L[j][i] = (this.data[j][i] - sum) / U[i][i];
                }
            }
        }

        return { L: new Matrix(L), U: new Matrix(U) };
    }
}

// Clase Vector
class Vector {
    constructor(data) {
        this.data = [...data];
        this.dim = data.length;
    }

    static add(a, b) {
        if (a.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para sumar');
        }
        const result = [];
        for (let i = 0; i < a.dim; i++) {
            result[i] = a.data[i] + b.data[i];
        }
        return new Vector(result);
    }

    static subtract(a, b) {
        if (a.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para restar');
        }
        const result = [];
        for (let i = 0; i < a.dim; i++) {
            result[i] = a.data[i] - b.data[i];
        }
        return new Vector(result);
    }

    static dotProduct(a, b) {
        if (a.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para el producto escalar');
        }
        let result = 0;
        for (let i = 0; i < a.dim; i++) {
            result += a.data[i] * b.data[i];
        }
        return result;
    }

    static crossProduct(a, b) {
        if (a.dim !== 3 || b.dim !== 3) {
            throw new Error('El producto vectorial solo está definido para vectores en 3D');
        }
        return new Vector([
            a.data[1] * b.data[2] - a.data[2] * b.data[1],
            a.data[2] * b.data[0] - a.data[0] * b.data[2],
            a.data[0] * b.data[1] - a.data[1] * b.data[0]
        ]);
    }

    magnitude() {
        let sum = 0;
        for (let i = 0; i < this.dim; i++) {
            sum += this.data[i] * this.data[i];
        }
        return Math.sqrt(sum);
    }

    angle(b) {
        if (this.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para calcular el ángulo');
        }
        const dot = Vector.dotProduct(this, b);
        const magA = this.magnitude();
        const magB = b.magnitude();
        return Math.acos(dot / (magA * magB));
    }

    normalize() {
        const mag = this.magnitude();
        const result = [];
        for (let i = 0; i < this.dim; i++) {
            result[i] = this.data[i] / mag;
        }
        return new Vector(result);
    }

    project(b) {
        if (this.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para la proyección');
        }
        const dot = Vector.dotProduct(this, b);
        const magB = b.magnitude();
        const scalar = dot / (magB * magB);
        const result = [];
        for (let i = 0; i < this.dim; i++) {
            result[i] = scalar * b.data[i];
        }
        return new Vector(result);
    }

    distance(b) {
        if (this.dim !== b.dim) {
            throw new Error('Los vectores deben tener la misma dimensión para calcular la distancia');
        }
        let sum = 0;
        for (let i = 0; i < this.dim; i++) {
            sum += Math.pow(this.data[i] - b.data[i], 2);
        }
        return Math.sqrt(sum);
    }

    tripleProduct(b, c) {
        if (this.dim !== 3 || b.dim !== 3 || c.dim !== 3) {
            throw new Error('El producto triple solo está definido para vectores en 3D');
        }
        return Vector.crossProduct(this, Vector.crossProduct(b, c));
    }

    scalarTripleProduct(b, c) {
        if (this.dim !== 3 || b.dim !== 3 || c.dim !== 3) {
            throw new Error('El producto triple escalar solo está definido para vectores en 3D');
        }
        return Vector.dotProduct(this, Vector.crossProduct(b, c));
    }

    perpendicular() {
        if (this.dim !== 3) {
            throw new Error('El vector perpendicular solo está definido para vectores en 3D');
        }
        // Encontrar un vector perpendicular arbitrario
        let result;
        if (this.data[0] !== 0 || this.data[1] !== 0) {
            result = new Vector([-this.data[1], this.data[0], 0]);
        } else {
            result = new Vector([0, -this.data[2], this.data[1]]);
        }
        return result.normalize();
    }
}

// Funciones para generar inputs de matrices y vectores
function generateMatrixInputs(matrixId, rows, cols) {
    const container = document.getElementById(`${matrixId}-inputs`);
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'matrix-input';
            input.id = `${matrixId}-${i}-${j}`;
            input.value = '0';
            container.appendChild(input);
        }
    }
}

function generateVectorInputs(vectorId, dim) {
    const container = document.getElementById(`${vectorId}-inputs`);
    container.innerHTML = '';
    container.style.gridTemplateColumns = '1fr';

    for (let i = 0; i < dim; i++) {
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'vector-input';
        input.id = `${vectorId}-${i}`;
        input.value = '0';
        container.appendChild(input);
    }
}

function generateEquationInputs(rows, cols) {
    const matrixContainer = document.getElementById('equations-inputs');
    matrixContainer.innerHTML = '';
    matrixContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    const constantsContainer = document.getElementById('constants-inputs');
    constantsContainer.innerHTML = '';
    constantsContainer.style.gridTemplateColumns = '1fr';

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'matrix-input';
            input.id = `equation-${i}-${j}`;
            input.value = '0';
            matrixContainer.appendChild(input);
        }
        const constantInput = document.createElement('input');
        constantInput.type = 'number';
        constantInput.className = 'vector-input';
        constantInput.id = `constant-${i}`;
        constantInput.value = '0';
        constantsContainer.appendChild(constantInput);
    }
}

// Funciones para llenar matrices y vectores con valores predefinidos
function fillMatrix(matrixId, type) {
    const rows = parseInt(document.getElementById(`${matrixId}-rows`).value);
    const cols = parseInt(document.getElementById(`${matrixId}-cols`).value);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const input = document.getElementById(`${matrixId}-${i}-${j}`);
            switch (type) {
                case 'identity':
                    input.value = i === j ? '1' : '0';
                    break;
                case 'zeros':
                    input.value = '0';
                    break;
                case 'ones':
                    input.value = '1';
                    break;
                case 'random':
                    input.value = Math.floor(Math.random() * 10);
                    break;
            }
        }
    }
}

function fillVector(vectorId, type) {
    const dim = parseInt(document.getElementById(`${vectorId}-dim`).value);

    for (let i = 0; i < dim; i++) {
        const input = document.getElementById(`${vectorId}-${i}`);
        switch (type) {
            case 'zeros':
                input.value = '0';
                break;
            case 'ones':
                input.value = '1';
                break;
            case 'random':
                input.value = Math.floor(Math.random() * 10);
                break;
            case 'unit':
                input.value = i === 0 ? '1' : '0';
                break;
        }
    }
}

function fillEquationMatrix(type) {
    const rows = parseInt(document.getElementById('equations-rows').value);
    const cols = parseInt(document.getElementById('equations-cols').value);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const input = document.getElementById(`equation-${i}-${j}`);
            switch (type) {
                case 'random':
                    input.value = Math.floor(Math.random() * 10);
                    break;
                case 'example':
                    if (rows === 3 && cols === 3) {
                        const examples = [
                            [2, 1, -1],
                            [-3, -1, 2],
                            [-2, 1, 2]
                        ];
                        input.value = examples[i][j];
                    } else {
                        input.value = Math.floor(Math.random() * 10);
                    }
                    break;
            }
        }
        const constantInput = document.getElementById(`constant-${i}`);
        switch (type) {
            case 'random':
                constantInput.value = Math.floor(Math.random() * 10);
                break;
            case 'example':
                if (rows === 3) {
                    const examples = [8, -11, -3];
                    constantInput.value = examples[i];
                } else {
                    constantInput.value = Math.floor(Math.random() * 10);
                }
                break;
        }
    }
}

function clearMatrix(matrixId) {
    const rows = parseInt(document.getElementById(`${matrixId}-rows`).value);
    const cols = parseInt(document.getElementById(`${matrixId}-cols`).value);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            document.getElementById(`${matrixId}-${i}-${j}`).value = '0';
        }
    }
}

function clearVector(vectorId) {
    const dim = parseInt(document.getElementById(`${vectorId}-dim`).value);

    for (let i = 0; i < dim; i++) {
        document.getElementById(`${vectorId}-${i}`).value = '0';
    }
}

function clearEquationMatrix() {
    const rows = parseInt(document.getElementById('equations-rows').value);
    const cols = parseInt(document.getElementById('equations-cols').value);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            document.getElementById(`equation-${i}-${j}`).value = '0';
        }
        document.getElementById(`constant-${i}`).value = '0';
    }
}

// Funciones para operaciones con matrices
function addMatrices() {
    try {
        const a = getMatrix('matrix-a');
        const b = getMatrix('matrix-b');
        const result = Matrix.add(a, b);
        displayMatrixResult('Suma de Matrices', result);
        displayMatrixProcedure('Suma de Matrices', a, b, result, 'add');
    } catch (error) {
        displayError(error.message);
    }
}

function subtractMatrices() {
    try {
        const a = getMatrix('matrix-a');
        const b = getMatrix('matrix-b');
        const result = Matrix.subtract(a, b);
        displayMatrixResult('Resta de Matrices', result);
        displayMatrixProcedure('Resta de Matrices', a, b, result, 'subtract');
    } catch (error) {
        displayError(error.message);
    }
}

function multiplyMatrices() {
    try {
        const a = getMatrix('matrix-a');
        const b = getMatrix('matrix-b');
        const result = Matrix.multiply(a, b);
        displayMatrixResult('Multiplicación de Matrices', result);
        displayMatrixProcedure('Multiplicación de Matrices', a, b, result, 'multiply');
    } catch (error) {
        displayError(error.message);
    }
}

function transposeMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.transpose();
        displayMatrixResult('Transpuesta de Matriz', result);
        displayMatrixProcedure('Transpuesta de Matriz', a, null, result, 'transpose');
    } catch (error) {
        displayError(error.message);
    }
}

function determinantMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.determinant();
        displayScalarResult('Determinante de Matriz', result);
        displayMatrixProcedure('Determinante de Matriz', a, null, null, 'determinant', result);
    } catch (error) {
        displayError(error.message);
    }
}

function scalarMultiply() {
    try {
        const a = getMatrix('matrix-a');
        const scalar = parseFloat(prompt('Ingrese el escalar:'));
        if (isNaN(scalar)) {
            throw new Error('El escalar debe ser un número válido');
        }
        const result = a.scalarMultiply(scalar);
        displayMatrixResult('Multiplicación por Escalar', result);
        displayMatrixProcedure('Multiplicación por Escalar', a, null, result, 'scalarMultiply', scalar);
    } catch (error) {
        displayError(error.message);
    }
}

function matrixPower() {
    try {
        const a = getMatrix('matrix-a');
        const n = parseInt(prompt('Ingrese la potencia:'));
        if (isNaN(n)) {
            throw new Error('La potencia debe ser un número entero válido');
        }
        const result = a.power(n);
        displayMatrixResult('Potencia de Matriz', result);
        displayMatrixProcedure('Potencia de Matriz', a, null, result, 'power', n);
    } catch (error) {
        displayError(error.message);
    }
}

function inverseMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.inverse();
        displayMatrixResult('Inversa de Matriz', result);
        displayMatrixProcedure('Inversa de Matriz', a, null, result, 'inverse');
    } catch (error) {
        displayError(error.message);
    }
}

function rankMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.rank();
        displayScalarResult('Rango de Matriz', result);
        displayMatrixProcedure('Rango de Matriz', a, null, null, 'rank', result);
    } catch (error) {
        displayError(error.message);
    }
}

function traceMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.trace();
        displayScalarResult('Traza de Matriz', result);
        displayMatrixProcedure('Traza de Matriz', a, null, null, 'trace', result);
    } catch (error) {
        displayError(error.message);
    }
}

function eigenvaluesMatrix() {
    try {
        const a = getMatrix('matrix-a');
        const result = a.eigenvalues();
        displayVectorResult('Eigenvalores de Matriz', result);
        displayMatrixProcedure('Eigenvalores de Matriz', a, null, null, 'eigenvalues', result);
    } catch (error) {
        displayError(error.message);
    }
}

function luDecomposition() {
    try {
        const a = getMatrix('matrix-a');
        const { L, U } = a.luDecomposition();
        displayMatrixResult('Descomposición LU - L', L);
        displayMatrixResult('Descomposición LU - U', U);
        displayMatrixProcedure('Descomposición LU', a, null, { L, U }, 'luDecomposition');
    } catch (error) {
        displayError(error.message);
    }
}

function determinantAx() {
    try {
        const a = getMatrix('matrix-a');
        const b = getVector('vector-b').data;
        const Aj = JSON.parse(JSON.stringify(a.data));
        for (let i = 0; i < a.rows; i++) {
            Aj[i][0] = b[i];
        }
        const detAj = new Matrix(Aj).determinant();
        displayScalarResult('Determinante de Ax', detAj);
        displayMatrixProcedure('Determinante de Ax', new Matrix(Aj), null, null, 'determinantAx', detAj);
    } catch (error) {
        displayError(error.message);
    }
}

function determinantAy() {
    try {
        const a = getMatrix('matrix-a');
        const b = getVector('vector-b').data;
        const Aj = JSON.parse(JSON.stringify(a.data));
        for (let i = 0; i < a.rows; i++) {
            Aj[i][1] = b[i];
        }
        const detAj = new Matrix(Aj).determinant();
        displayScalarResult('Determinante de Ay', detAj);
        displayMatrixProcedure('Determinante de Ay', new Matrix(Aj), null, null, 'determinantAy', detAj);
    } catch (error) {
        displayError(error.message);
    }
}

function determinantAz() {
    try {
        const a = getMatrix('matrix-a');
        const b = getVector('vector-b').data;
        const Aj = JSON.parse(JSON.stringify(a.data));
        for (let i = 0; i < a.rows; i++) {
            Aj[i][2] = b[i];
        }
        const detAj = new Matrix(Aj).determinant();
        displayScalarResult('Determinante de Az', detAj);
        displayMatrixProcedure('Determinante de Az', new Matrix(Aj), null, null, 'determinantAz', detAj);
    } catch (error) {
        displayError(error.message);
    }
}

// Funciones para operaciones con vectores
function addVectors() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = Vector.add(a, b);
        displayVectorResult('Suma de Vectores', result);
        displayVectorProcedure('Suma de Vectores', a, b, result, 'add');
    } catch (error) {
        displayError(error.message);
    }
}

function subtractVectors() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = Vector.subtract(a, b);
        displayVectorResult('Resta de Vectores', result);
        displayVectorProcedure('Resta de Vectores', a, b, result, 'subtract');
    } catch (error) {
        displayError(error.message);
    }
}

function dotProduct() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = Vector.dotProduct(a, b);
        displayScalarResult('Producto Escalar', result);
        displayVectorProcedure('Producto Escalar', a, b, null, 'dotProduct', result);
    } catch (error) {
        displayError(error.message);
    }
}

function crossProduct() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = Vector.crossProduct(a, b);
        displayVectorResult('Producto Vectorial', result);
        displayVectorProcedure('Producto Vectorial', a, b, result, 'crossProduct');
    } catch (error) {
        displayError(error.message);
    }
}

function vectorMagnitude() {
    try {
        const a = getVector('vector-a');
        const result = a.magnitude();
        displayScalarResult('Magnitud de Vector', result);
        displayVectorProcedure('Magnitud de Vector', a, null, null, 'magnitude', result);
    } catch (error) {
        displayError(error.message);
    }
}

function vectorAngle() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = a.angle(b);
        displayScalarResult('Ángulo entre Vectores (radianes)', result);
        displayVectorProcedure('Ángulo entre Vectores', a, b, null, 'angle', result);
    } catch (error) {
        displayError(error.message);
    }
}

function normalizeVector() {
    try {
        const a = getVector('vector-a');
        const result = a.normalize();
        displayVectorResult('Vector Normalizado', result);
        displayVectorProcedure('Vector Normalizado', a, null, result, 'normalize');
    } catch (error) {
        displayError(error.message);
    }
}

function projectVector() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = a.project(b);
        displayVectorResult('Proyección de Vector', result);
        displayVectorProcedure('Proyección de Vector', a, b, result, 'project');
    } catch (error) {
        displayError(error.message);
    }
}

function distanceBetweenVectors() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const result = a.distance(b);
        displayScalarResult('Distancia entre Vectores', result);
        displayVectorProcedure('Distancia entre Vectores', a, b, null, 'distance', result);
    } catch (error) {
        displayError(error.message);
    }
}

function vectorTripleProduct() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const c = getVector('vector-b'); // Usando el mismo vector para simplificar
        const result = a.tripleProduct(b, c);
        displayVectorResult('Producto Triple Vectorial', result);
        displayVectorProcedure('Producto Triple Vectorial', a, b, result, 'tripleProduct', c);
    } catch (error) {
        displayError(error.message);
    }
}

function scalarTripleProduct() {
    try {
        const a = getVector('vector-a');
        const b = getVector('vector-b');
        const c = getVector('vector-b'); // Usando el mismo vector para simplificar
        const result = a.scalarTripleProduct(b, c);
        displayScalarResult('Producto Triple Escalar', result);
        displayVectorProcedure('Producto Triple Escalar', a, b, null, 'scalarTripleProduct', { c, result });
    } catch (error) {
        displayError(error.message);
    }
}

function vectorPerpendicular() {
    try {
        const a = getVector('vector-a');
        const result = a.perpendicular();
        displayVectorResult('Vector Perpendicular', result);
        displayVectorProcedure('Vector Perpendicular', a, null, result, 'perpendicular');
    } catch (error) {
        displayError(error.message);
    }
}

// Funciones para operaciones con sistemas de ecuaciones
function solveSystemGauss() {
    try {
        const { A, b } = getEquationSystem();
        const result = solveGauss(A, b);
        displayVectorResult('Solución por Método de Gauss', result);
        displayEquationProcedure('Método de Gauss', A, b, result, 'gauss');
    } catch (error) {
        displayError(error.message);
    }
}

function solveSystemCramer() {
    try {
        const { A, b } = getEquationSystem();
        const result = solveCramer(A, b);
        displayVectorResult('Solución por Regla de Cramer', result);
        displayEquationProcedure('Regla de Cramer', A, b, result, 'cramer');
    } catch (error) {
        displayError(error.message);
    }
}

function solveSystemMatrix() {
    try {
        const { A, b } = getEquationSystem();
        const result = solveMatrix(A, b);
        displayVectorResult('Solución por Método Matricial', result);
        displayEquationProcedure('Método Matricial', A, b, result, 'matrix');
    } catch (error) {
        displayError(error.message);
    }
}

function systemDeterminant() {
    try {
        const { A } = getEquationSystem();
        const result = new Matrix(A).determinant();
        displayScalarResult('Determinante del Sistema', result);
        displayEquationProcedure('Determinante del Sistema', A, null, null, 'determinant', result);
    } catch (error) {
        displayError(error.message);
    }
}

function systemRank() {
    try {
        const { A } = getEquationSystem();
        const result = new Matrix(A).rank();
        displayScalarResult('Rango del Sistema', result);
        displayEquationProcedure('Rango del Sistema', A, null, null, 'rank', result);
    } catch (error) {
        displayError(error.message);
    }
}

function systemConsistency() {
    try {
        const { A, b } = getEquationSystem();
        const result = checkConsistency(A, b);
        displayScalarResult('Consistencia del Sistema', result ? 'Consistente' : 'Inconsistente');
        displayEquationProcedure('Consistencia del Sistema', A, b, null, 'consistency', result);
    } catch (error) {
        displayError(error.message);
    }
}

// Funciones para operaciones con calculadora
function evaluateExpression() {
    try {
        const expression = document.getElementById('expression-input').value;
        const result = math.evaluate(expression);
        displayScalarResult('Resultado de Expresión', result);
        displayCalculatorProcedure('Evaluar Expresión', expression, result);
    } catch (error) {
        displayError(error.message);
    }
}

function calculateFactorial() {
    try {
        const n = parseInt(prompt('Ingrese un número entero:'));
        if (isNaN(n) || n < 0) {
            throw new Error('Debe ingresar un número entero no negativo');
        }
        const result = factorial(n);
        displayScalarResult('Factorial', result);
        displayCalculatorProcedure('Factorial', n, result);
    } catch (error) {
        displayError(error.message);
    }
}

function calculateCombination() {
    try {
        const n = parseInt(prompt('Ingrese n:'));
        const r = parseInt(prompt('Ingrese r:'));
        if (isNaN(n) || isNaN(r) || n < 0 || r < 0 || r > n) {
            throw new Error('Valores inválidos para n y r');
        }
        const result = combination(n, r);
        displayScalarResult('Combinación C(n,r)', result);
        displayCalculatorProcedure('Combinación C(n,r)', { n, r }, result);
    } catch (error) {
        displayError(error.message);
    }
}

function calculatePermutation() {
    try {
        const n = parseInt(prompt('Ingrese n:'));
        const r = parseInt(prompt('Ingrese r:'));
        if (isNaN(n) || isNaN(r) || n < 0 || r < 0 || r > n) {
            throw new Error('Valores inválidos para n y r');
        }
        const result = permutation(n, r);
        displayScalarResult('Permutación P(n,r)', result);
        displayCalculatorProcedure('Permutación P(n,r)', { n, r }, result);
    } catch (error) {
        displayError(error.message);
    }
}

function calculateGCD() {
    try {
        const a = parseInt(prompt('Ingrese el primer número:'));
        const b = parseInt(prompt('Ingrese el segundo número:'));
        if (isNaN(a) || isNaN(b)) {
            throw new Error('Debe ingresar números válidos');
        }
        const result = gcd(a, b);
        displayScalarResult('MCD', result);
        displayCalculatorProcedure('MCD', { a, b }, result);
    } catch (error) {
        displayError(error.message);
    }
}

function calculateLCM() {
    try {
        const a = parseInt(prompt('Ingrese el primer número:'));
        const b = parseInt(prompt('Ingrese el segundo número:'));
        if (isNaN(a) || isNaN(b)) {
            throw new Error('Debe ingresar números válidos');
        }
        const result = lcm(a, b);
        displayScalarResult('MCM', result);
        displayCalculatorProcedure('MCM', { a, b }, result);
    } catch (error) {
        displayError(error.message);
    }
}

function convertToRadians() {
    try {
        const degrees = parseFloat(document.getElementById('degrees-input').value);
        if (isNaN(degrees)) {
            throw new Error('Debe ingresar un número válido');
        }
        const result = degrees * Math.PI / 180;
        document.getElementById('radians-input').value = result.toFixed(4);
        displayScalarResult('Conversión a Radianes', result);
        displayCalculatorProcedure('Conversión a Radianes', degrees, result);
    } catch (error) {
        displayError(error.message);
    }
}

function convertToDegrees() {
    try {
        const radians = parseFloat(document.getElementById('radians-input').value);
        if (isNaN(radians)) {
            throw new Error('Debe ingresar un número válido');
        }
        const result = radians * 180 / Math.PI;
        document.getElementById('degrees-input').value = result.toFixed(4);
        displayScalarResult('Conversión a Grados', result);
        displayCalculatorProcedure('Conversión a Grados', radians, result);
    } catch (error) {
        displayError(error.message);
    }
}

function insertFunction(func) {
    const expressionInput = document.getElementById('expression-input');
    expressionInput.value += func;
}

function clearExpression() {
    document.getElementById('expression-input').value = '';
}

// Funciones auxiliares para obtener matrices y vectores
function getMatrix(matrixId) {
    const rows = parseInt(document.getElementById(`${matrixId}-rows`).value);
    const cols = parseInt(document.getElementById(`${matrixId}-cols`).value);
    const data = [];

    for (let i = 0; i < rows; i++) {
        data[i] = [];
        for (let j = 0; j < cols; j++) {
            const value = parseFloat(document.getElementById(`${matrixId}-${i}-${j}`).value);
            if (isNaN(value)) {
                throw new Error(`Valor inválido en la posición (${i+1}, ${j+1})`);
            }
            data[i][j] = value;
        }
    }

    return new Matrix(data);
}

function getVector(vectorId) {
    const dim = parseInt(document.getElementById(`${vectorId}-dim`).value);
    const data = [];

    for (let i = 0; i < dim; i++) {
        const value = parseFloat(document.getElementById(`${vectorId}-${i}`).value);
        if (isNaN(value)) {
            throw new Error(`Valor inválido en la posición ${i+1}`);
        }
        data[i] = value;
    }

    return new Vector(data);
}

function getEquationSystem() {
    const rows = parseInt(document.getElementById('equations-rows').value);
    const cols = parseInt(document.getElementById('equations-cols').value);
    const A = [];
    const b = [];

    for (let i = 0; i < rows; i++) {
        A[i] = [];
        for (let j = 0; j < cols; j++) {
            const value = parseFloat(document.getElementById(`equation-${i}-${j}`).value);
            if (isNaN(value)) {
                throw new Error(`Valor inválido en la posición (${i+1}, ${j+1})`);
            }
            A[i][j] = value;
        }
        const constantValue = parseFloat(document.getElementById(`constant-${i}`).value);
        if (isNaN(constantValue)) {
            throw new Error(`Valor inválido en el término independiente ${i+1}`);
        }
        b[i] = constantValue;
    }

    return { A, b };
}

// Funciones para resolver sistemas de ecuaciones
function solveGauss(A, b) {
    const n = A.length;
    const augmented = [];

    for (let i = 0; i < n; i++) {
        augmented[i] = [...A[i], b[i]];
    }

    for (let i = 0; i < n; i++) {
        // Búsqueda de pivote
        let max = Math.abs(augmented[i][i]);
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > max) {
                max = Math.abs(augmented[k][i]);
                maxRow = k;
            }
        }

        // Intercambio de filas
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

        // Eliminación hacia adelante
        for (let k = i + 1; k < n; k++) {
            const factor = augmented[k][i] / augmented[i][i];
            for (let j = 0; j <= n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }

    // Sustitución hacia atrás
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = augmented[i][n];
        for (let j = i + 1; j < n; j++) {
            x[i] -= augmented[i][j] * x[j];
        }
        x[i] /= augmented[i][i];
    }

    return x;
}

function solveCramer(A, b) {
    const n = A.length;
    const detA = new Matrix(A).determinant();
    if (detA === 0) {
        throw new Error('El sistema no tiene solución única (determinante es cero)');
    }

    const x = new Array(n);
    for (let j = 0; j < n; j++) {
        const Aj = JSON.parse(JSON.stringify(A));
        for (let i = 0; i < n; i++) {
            Aj[i][j] = b[i];
        }
        const detAj = new Matrix(Aj).determinant();
        x[j] = detAj / detA;
    }

    return x;
}

function solveMatrix(A, b) {
    const matrixA = new Matrix(A);
    const invA = matrixA.inverse();
    const vectorB = new Vector(b);
    const result = invA.multiply(new Matrix([vectorB.data])).data[0];
    return result;
}

function checkConsistency(A, b) {
    const augmented = [];
    const n = A.length;
    const m = A[0].length;

    for (let i = 0; i < n; i++) {
        augmented[i] = [...A[i], b[i]];
    }

    let rankA = 0;
    let rankAugmented = 0;

    // Calcular rango de A
    const tempA = JSON.parse(JSON.stringify(A));
    for (let i = 0; i < n; i++) {
        let allZeros = true;
        for (let j = 0; j < m; j++) {
            if (tempA[i][j] !== 0) {
                allZeros = false;
                break;
            }
        }
        if (!allZeros) {
            rankA++;
        }
    }

    // Calcular rango de la matriz aumentada
    for (let i = 0; i < n; i++) {
        let allZeros = true;
        for (let j = 0; j <= m; j++) {
            if (augmented[i][j] !== 0) {
                allZeros = false;
                break;
            }
        }
        if (!allZeros) {
            rankAugmented++;
        }
    }

    return rankA === rankAugmented;
}

// Funciones matemáticas auxiliares
function factorial(n) {
    if (n === 0 || n === 1) {
        return 1;
    }
    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

function combination(n, r) {
    return factorial(n) / (factorial(r) * factorial(n - r));
}

function permutation(n, r) {
    return factorial(n) / factorial(n - r);
}

function gcd(a, b) {
    if (b === 0) {
        return a;
    }
    return gcd(b, a % b);
}

function lcm(a, b) {
    return (a * b) / gcd(a, b);
}

// Funciones para mostrar resultados
function displayMatrixResult(title, matrix) {
    const resultDiv = document.getElementById('matrix-result');
    resultDiv.innerHTML = `
        <div class="result-title">${title}</div>
        <div class="matrix-display">
            ${matrixToString(matrix)}
        </div>
        <button class="copy-btn" onclick="copyToClipboard('${matrixToString(matrix)}')">Copiar</button>
    `;
}

function displayVectorResult(title, vector) {
    const resultDiv = document.getElementById(vector instanceof Vector ? 'vector-result' : 'equation-result');
    resultDiv.innerHTML = `
        <div class="result-title">${title}</div>
        <div class="vector-display">
            ${vectorToString(vector)}
        </div>
        <button class="copy-btn" onclick="copyToClipboard('${vectorToString(vector)}')">Copiar</button>
    `;
}

function displayScalarResult(title, value) {
    const resultDiv = document.getElementById('matrix-result');
    resultDiv.innerHTML = `
        <div class="result-title">${title}</div>
        <div class="result-content">
            ${value}
        </div>
        <button class="copy-btn" onclick="copyToClipboard('${value}')">Copiar</button>
    `;
}

function displayError(message) {
    const resultDiv = document.getElementById('matrix-result');
    resultDiv.innerHTML = `
        <div class="error-message">
            Error: ${message}
        </div>
    `;
}

function displayMatrixProcedure(title, a, b, result, operation, extra) {
    const procedureDiv = document.getElementById('matrix-procedure');
    let procedureHTML = `<div class="procedure-title">${title}</div>`;

    if (a) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Matriz A:</strong>
                    <div class="matrix-display">
                        ${matrixToString(a)}
                    </div>
                </div>
            </div>
        `;
    }

    if (b) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Matriz B:</strong>
                    <div class="matrix-display">
                        ${matrixToString(b)}
                    </div>
                </div>
            </div>
        `;
    }

    if (operation === 'add') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Sumar los elementos correspondientes de A y B.
                    <div class="matrix-display">
                        ${matrixToString(a)} + ${matrixToString(b)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'subtract') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Restar los elementos correspondientes de A y B.
                    <div class="matrix-display">
                        ${matrixToString(a)} - ${matrixToString(b)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'multiply') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Multiplicar cada fila de A por cada columna de B y sumar los productos.
                    <div class="matrix-display">
                        ${matrixToString(a)} × ${matrixToString(b)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'transpose') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Intercambiar filas por columnas.
                    <div class="matrix-display">
                        Transpuesta de ${matrixToString(a)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'determinant') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante usando expansión por cofactores.
                    <div class="matrix-display">
                        Determinante de ${matrixToString(a)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'scalarMultiply') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Multiplicar cada elemento de A por el escalar ${extra}.
                    <div class="matrix-display">
                        ${extra} × ${matrixToString(a)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'power') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Multiplicar la matriz por sí misma ${extra} veces.
                    <div class="matrix-display">
                        ${matrixToString(a)}^${extra} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'inverse') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular la matriz adjunta y dividir por el determinante.
                    <div class="matrix-display">
                        Inversa de ${matrixToString(a)} = ${matrixToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'rank') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Reducir la matriz a su forma escalonada por filas.
                    <div class="matrix-display">
                        Rango de ${matrixToString(a)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'trace') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Sumar los elementos de la diagonal principal.
                    <div class="matrix-display">
                        Traza de ${matrixToString(a)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'eigenvalues') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Resolver la ecuación característica det(A - λI) = 0.
                    <div class="matrix-display">
                        Eigenvalores de ${matrixToString(a)} = [${extra.join(', ')}]
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'luDecomposition') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Descomponer A en una matriz triangular inferior L y una matriz triangular superior U.
                    <div class="matrix-display">
                        L = ${matrixToString(extra.L)}
                    </div>
                    <div class="matrix-display">
                        U = ${matrixToString(extra.U)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'determinantAx') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante de Ax.
                    <div class="matrix-display">
                        Determinante de Ax = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'determinantAy') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante de Ay.
                    <div class="matrix-display">
                        Determinante de Ay = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'determinantAz') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante de Az.
                    <div class="matrix-display">
                        Determinante de Az = ${extra}
                    </div>
                </div>
            </div>
        `;
    }

    procedureDiv.innerHTML = procedureHTML;
}

function displayVectorProcedure(title, a, b, result, operation, extra) {
    const procedureDiv = document.getElementById('vector-procedure');
    let procedureHTML = `<div class="procedure-title">${title}</div>`;

    if (a) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Vector A:</strong>
                    <div class="vector-display">
                        ${vectorToString(a)}
                    </div>
                </div>
            </div>
        `;
    }

    if (b) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Vector B:</strong>
                    <div class="vector-display">
                        ${vectorToString(b)}
                    </div>
                </div>
            </div>
        `;
    }

    if (operation === 'add') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Sumar los componentes correspondientes de A y B.
                    <div class="vector-display">
                        ${vectorToString(a)} + ${vectorToString(b)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'subtract') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Restar los componentes correspondientes de A y B.
                    <div class="vector-display">
                        ${vectorToString(a)} - ${vectorToString(b)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'dotProduct') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Multiplicar los componentes correspondientes y sumar los productos.
                    <div class="vector-display">
                        ${vectorToString(a)} · ${vectorToString(b)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'crossProduct') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el producto vectorial usando el determinante.
                    <div class="vector-display">
                        ${vectorToString(a)} × ${vectorToString(b)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'magnitude') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular la raíz cuadrada de la suma de los cuadrados de los componentes.
                    <div class="vector-display">
                        Magnitud de ${vectorToString(a)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'angle') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Usar el producto escalar y las magnitudes para calcular el ángulo.
                    <div class="vector-display">
                        Ángulo entre ${vectorToString(a)} y ${vectorToString(b)} = ${extra} radianes
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'normalize') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Dividir cada componente por la magnitud del vector.
                    <div class="vector-display">
                        Normalización de ${vectorToString(a)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'project') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular la proyección de A sobre B.
                    <div class="vector-display">
                        Proyección de ${vectorToString(a)} sobre ${vectorToString(b)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'distance') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular la distancia euclidiana entre A y B.
                    <div class="vector-display">
                        Distancia entre ${vectorToString(a)} y ${vectorToString(b)} = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'tripleProduct') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el producto triple vectorial A × (B × C).
                    <div class="vector-display">
                        ${vectorToString(a)} × (${vectorToString(b)} × ${vectorToString(extra)}) = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'scalarTripleProduct') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el producto triple escalar A · (B × C).
                    <div class="vector-display">
                        ${vectorToString(a)} · (${vectorToString(b)} × ${vectorToString(extra.c)}) = ${extra.result}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'perpendicular') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Encontrar un vector perpendicular a A.
                    <div class="vector-display">
                        Vector perpendicular a ${vectorToString(a)} = ${vectorToString(result)}
                    </div>
                </div>
            </div>
        `;
    }

    procedureDiv.innerHTML = procedureHTML;
}

function displayEquationProcedure(title, A, b, result, operation, extra) {
    const procedureDiv = document.getElementById('equation-procedure');
    let procedureHTML = `<div class="procedure-title">${title}</div>`;

    if (A) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Matriz de Coeficientes:</strong>
                    <div class="matrix-display">
                        ${matrixToString(new Matrix(A))}
                    </div>
                </div>
            </div>
        `;
    }

    if (b) {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Vector de Términos Independientes:</strong>
                    <div class="vector-display">
                        ${vectorToString(new Vector(b))}
                    </div>
                </div>
            </div>
        `;
    }

    if (operation === 'gauss') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Reducir la matriz aumentada a su forma escalonada por filas.
                    <div class="vector-display">
                        Solución: [${result.join(', ')}]
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'cramer') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante de A y los determinantes de las matrices modificadas.
                    <div class="vector-display">
                        Solución: [${result.join(', ')}]
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'matrix') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Multiplicar la inversa de A por el vector de términos independientes.
                    <div class="vector-display">
                        Solución: [${result.join(', ')}]
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'determinant') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Calcular el determinante de la matriz de coeficientes.
                    <div class="matrix-display">
                        Determinante = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'rank') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Reducir la matriz a su forma escalonada por filas.
                    <div class="matrix-display">
                        Rango = ${extra}
                    </div>
                </div>
            </div>
        `;
    } else if (operation === 'consistency') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Paso 1:</strong> Comparar el rango de la matriz de coeficientes con el de la matriz aumentada.
                    <div class="matrix-display">
                        El sistema es ${extra ? 'consistente' : 'inconsistente'}
                    </div>
                </div>
            </div>
        `;
    }

    procedureDiv.innerHTML = procedureHTML;
}

function displayCalculatorProcedure(title, input, result) {
    const procedureDiv = document.getElementById('calculator-procedure');
    let procedureHTML = `<div class="procedure-title">${title}</div>`;

    if (typeof input === 'string') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Expresión:</strong>
                    <div class="result-content">
                        ${input}
                    </div>
                </div>
            </div>
        `;
    } else if (typeof input === 'object') {
        procedureHTML += `
            <div class="steps-visualization">
                <div class="procedure-steps">
                    <strong>Entrada:</strong>
                    <div class="result-content">
                        ${JSON.stringify(input)}
                    </div>
                </div>
            </div>
        `;
    }

    procedureHTML += `
        <div class="steps-visualization">
            <div class="procedure-steps">
                <strong>Resultado:</strong>
                <div class="result-content">
                    ${result}
                </div>
            </div>
        </div>
    `;

    procedureDiv.innerHTML = procedureHTML;
}

function matrixToString(matrix) {
    let str = '';
    for (let i = 0; i < matrix.rows; i++) {
        str += matrix.data[i].join(' ') + '<br>';
    }
    return str;
}

function vectorToString(vector) {
    if (vector instanceof Vector) {
        return vector.data.join(' ');
    } else if (Array.isArray(vector)) {
        return vector.join(' ');
    } else {
        return vector;
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Copiado al portapapeles');
    }).catch(err => {
        console.error('Error al copiar: ', err);
    });
}

// Función para mostrar secciones
function showSection(sectionId) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.remove('active');
    });

    const navBtns = document.querySelectorAll('.nav-btn');
    navBtns.forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(sectionId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    generateMatrixInputs('matrix-a', 3, 3);
    generateMatrixInputs('matrix-b', 3, 3);
    generateVectorInputs('vector-a', 3);
    generateVectorInputs('vector-b', 3);
    generateEquationInputs(3, 3);
});
