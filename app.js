class UniversalApproximator {
    constructor() {
        this.model = null;
        this.chart = null;
        this.isTraining = false;
        this.xData = null;
        this.yData = null;
        this.init();
    }
    init() {
        this.setupEventListeners();
        this.setupChart();
        this.generateData();
    }
    setupEventListeners() {
        document.getElementById('neurons').addEventListener('input', (e) => {
            document.getElementById('neurons-value').textContent = e.target.value;
        });
        document.getElementById('learning-rate').addEventListener('input', (e) => {
            document.getElementById('learning-rate-value').textContent = e.target.value;
        });
        document.getElementById('epochs').addEventListener('input', (e) => {
            document.getElementById('epochs-value').textContent = e.target.value;
        });
        document.getElementById('function-input').addEventListener('input', () => {
            this.generateData();
        });
        document.getElementById('train-btn').addEventListener('click', () => {
            this.trainModel();
        });
    }
    createModel() {
        const numNeurons = parseInt(document.getElementById('neurons').value);
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [1],
                    units: numNeurons,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'linear'
                })
            ]
        });
        const lr = parseFloat(document.getElementById('learning-rate').value);
        model.compile({
            optimizer: tf.train.adam(lr),
            loss: 'meanSquaredError'
        });
        return model;
    }
    generateData() {
        const func = document.getElementById('function-input').value;
        const expr = math.parse(func);
        const compiled = expr.compile();
        const xTrain = [];
        const yTrain = [];
        for (let x = -5; x <= 5; x += 0.05) {
            xTrain.push(x);
            const y = compiled.evaluate({ x: x });
            yTrain.push(y);
        }
        this.xData = tf.tensor2d(xTrain, [xTrain.length, 1]);
        this.yData = tf.tensor2d(yTrain, [yTrain.length, 1]);
    }
    async trainModel() {
        this.model = this.createModel();
        const epochs = parseInt(document.getElementById('epochs').value);
        await this.model.fit(this.xData, this.yData, { epochs: epochs });

        await this.visualizeResults();
    }
    setupChart() {
        const ctx = document.getElementById('main-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                    }
                },
                scales: {
                    x: { 
                        type: 'linear',
                        title: { 
                            display: true, 
                            text: 'x',
                            color: '#f4f4f5',
                            font: { size: 16, weight: '600' }
                        },
                        ticks: { 
                            color: '#d4d4d8',
                            font: { size: 14 }
                        },
                        grid: { 
                            color: '#404040',
                            lineWidth: 1
                        }
                    },
                    y: { 
                        type: 'linear',
                        title: { 
                            display: true, 
                            text: 'f(x)',
                            color: '#f4f4f5',
                            font: { size: 16, weight: '600' }
                        },
                        ticks: { 
                            color: '#d4d4d8',
                            font: { size: 14 }
                        },
                        grid: { 
                            color: '#404040',
                            lineWidth: 1
                        }
                    }
                }
            }
        });
    }
    async visualizeResults() {
        const preds = this.model.predict(this.xData);
        const hiddenModel = tf.model({
            inputs: this.model.input,
            outputs: this.model.layers[0].output
        });
        const hiddenOutputs = hiddenModel.predict(this.xData);
        const outputWeights = this.model.layers[1].getWeights()[0];
        const predData = await preds.data();
        const hiddenData = await hiddenOutputs.data();
        const weightData = await outputWeights.data();
        const xValues = await this.xData.data();
        const yTargetData = await this.yData.data();
        const numNeurons = parseInt(document.getElementById('neurons').value);
        const neuronContributions = [];
        for (let neuron = 0; neuron < numNeurons; neuron++) {
            const contributions = [];
            for (let i = 0; i < xValues.length; i++) {
                const hiddenValue = hiddenData[i * numNeurons +
                    neuron];
                const weight = weightData[neuron];
                contributions.push(hiddenValue * weight);
            }
            neuronContributions.push(contributions);
        }

      const datasets = [];

      datasets.push({
            label: 'Target Function',
            data: Array.from(xValues).map((x, i) => ({x, y:
        yTargetData[i]})),
            borderColor: 'red',
            borderDash: [5, 5]
        });
      datasets.push({
            label: 'Neural Network',
            data: Array.from(xValues).map((x, i) => ({x, y:
        predData[i]})),
            borderColor: 'blue'
        });

      this.chart.data.datasets = datasets;
        this.chart.update();
    }
}
document.addEventListener('DOMContentLoaded', () => {
    new UniversalApproximator();
});
