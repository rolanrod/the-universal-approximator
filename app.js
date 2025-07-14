class UniversalApproximator {
    constructor() {
        this.model = null;
        this.chart = null;
        this.isTraining = false;
        this.xData = null;
        this.yData = null;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupChart();
        this.setupNetworkCanvas();
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

        document.getElementById('zoom-in').addEventListener('click', () => {
            this.zoom = Math.min(this.zoom * 1.2, 3);
            if (this.model) this.drawNetwork();
        });

        document.getElementById('zoom-out').addEventListener('click', () => {
            this.zoom = Math.max(this.zoom * 0.8, 0.5);
            if (this.model) this.drawNetwork();
        });

        document.getElementById('reset-view').addEventListener('click', () => {
            this.zoom = 1;
            this.panX = 0;
            this.panY = 0;
            if (this.model) this.drawNetwork();
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
        document.getElementById('loading-overlay').classList.remove('hidden');
        this.model = this.createModel();
        const epochs = parseInt(document.getElementById('epochs').value);
        await this.model.fit(this.xData, this.yData, { epochs: epochs });

        this.drawNetwork();
        await this.visualizeResults();
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    setupNetworkCanvas() {
      const canvas = document.getElementById('network-canvas');
      const ctx = canvas.getContext('2d');

      const dpr = window.devicePixelRatio || 1;
      canvas.width = 400 * dpr;
      canvas.height = 500 * dpr;
      canvas.style.width = '400px';
      canvas.style.height = '500px';
      ctx.scale(dpr, dpr);

      canvas.addEventListener('wheel', (e) => {
          e.preventDefault();
          const delta = e.deltaY > 0 ? 0.9 : 1.1;
          this.zoom = Math.min(Math.max(0.5, this.zoom * delta), 3);
          if (this.model) this.drawNetwork(); 
      });

      let isDragging = false;
      let lastX, lastY;

      canvas.addEventListener('mousedown', (e) => {
          isDragging = true;
          lastX = e.offsetX;
          lastY = e.offsetY;
      });

      canvas.addEventListener('mousemove', (e) => {
          if (isDragging) {
              this.panX += (e.offsetX - lastX) / this.zoom;
              this.panY += (e.offsetY - lastY) / this.zoom;
              lastX = e.offsetX;
              lastY = e.offsetY;
              if (this.model) this.drawNetwork();
          }
      });

      canvas.addEventListener('mouseup', () => {
          isDragging = false;
      });

      canvas.addEventListener('mouseleave', () => {
          isDragging = false;
      });

      this.networkCtx = ctx;
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

    drawNode(ctx, x, y, color, label) {
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 2;
        ctx.stroke();

        if (label) {
            ctx.fillStyle = '#f4f4f5';
            ctx.font = '12px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(label, x, y - 20);
        }
    }

    drawNetwork() {
        const ctx = this.networkCtx;
        const canvas = document.getElementById('network-canvas');
        
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);

        const inputX = 50;
        const outputX = 350;
        const hiddenX = 200;
        const centerY = 250;

        const numNeurons = parseInt(document.getElementById('neurons').value);
        const neuronSpacing = Math.max(400 / numNeurons, 12)

        ctx.strokeStyle = '#404040';
        ctx.lineWidth = 1;
        
        for (let i = 0; i < numNeurons; i++) {
            const y = centerY + (i - numNeurons/2) * neuronSpacing;
            ctx.beginPath();
            ctx.moveTo(inputX + 12, centerY);
            ctx.lineTo(hiddenX - 12, y);
            ctx.stroke();
        }
        
        for (let i = 0; i < numNeurons; i++) {
            const y = centerY + (i - numNeurons/2) * neuronSpacing;
            ctx.beginPath();
            ctx.moveTo(hiddenX + 12, y);
            ctx.lineTo(outputX - 12, centerY);
            ctx.stroke();
        }

        this.drawNode(ctx, inputX, centerY, '#06b6d4', 'Input');

        for (let i = 0; i < numNeurons; i++){
            const y = centerY + (i - numNeurons/2) * neuronSpacing;
            this.drawNode(ctx, hiddenX, y, '#404040', '');
        }

        this.drawNode(ctx, outputX, centerY, '#06b6d4', 'Output');
        
        ctx.restore();
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
