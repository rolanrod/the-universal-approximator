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
        this.neuronData = null;
        this.animationRunning = false;
        this.chartZoom = 1;
        this.chartPanX = 0;
        this.chartPanY = 0;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupChart();
        this.setupNetworkCanvas();
        this.setupChartZoom();
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

        // Chart zoom controls
        document.getElementById('chart-zoom-in').addEventListener('click', () => {
            this.chartZoom = Math.min(this.chartZoom * 1.2, 5);
            this.updateChartZoom();
        });

        document.getElementById('chart-zoom-out').addEventListener('click', () => {
            this.chartZoom = Math.max(this.chartZoom * 0.8, 0.2);
            this.updateChartZoom();
        });

        document.getElementById('chart-reset').addEventListener('click', () => {
            this.chartZoom = 1;
            this.chartPanX = 0;
            this.chartPanY = 0;
            this.updateChartZoom();
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
                    },
                    legend: {
                        display: false
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

    drawNetwork(highlightStart = -1, highlightEnd = -1) {
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
            
            const isHighlighted = (highlightStart >= 0 && i >= highlightStart && i < highlightEnd);
            const color = isHighlighted ? '#06b6d4' : '#404040';
            
            this.drawNode(ctx, hiddenX, y, color, '');
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

        this.neuronData = {
            predictions: predData,
            hiddenActivations: hiddenData,
            weights: weightData,
            xValues: xValues,
            yTarget: yTargetData,
            numNeurons: numNeurons,
        };

        this.animateReLUs();
        preds.dispose();
        hiddenOutputs.dispose();
        outputWeights.dispose();
        hiddenModel.dispose();

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

    async animateReLUs() {
        this.animationRunning = true;
        this.showTargetFunction();
        const groupSize = Math.max(1, Math.floor(this.neuronData.numNeurons / 10));
        const numGroups = Math.ceil(this.neuronData.numNeurons / groupSize);

        let cumContributions = new Array(this.neuronData.xValues.length).fill(0);
        for (let group = 0; group < numGroups; group++) {
            if (!this.animationRunning) break;
            const startNeuron = group * groupSize;
            const endNeuron = Math.min(startNeuron + groupSize, this.neuronData.numNeurons);

            this.highlightNeuronGroup(startNeuron, endNeuron);
            this.addGroupContributions(cumContributions, startNeuron, endNeuron);

            await new Promise(resolve => setTimeout(resolve, 800));
        }
        this.showFinalResult();
        this.animationRunning = false;
    }

    showTargetFunction() {
      const datasets = [{
          label: 'Target Function',
          data: Array.from(this.neuronData.xValues).map((x, i) => ({
              x,
              y: this.neuronData.yTarget[i]
          })),
          borderColor: '#ef4444',
          borderWidth: 3,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
      }];

      this.chart.data.datasets = datasets;
      this.chart.update('none');
  }

    addGroupContributions(cumulativeContributions, startNeuron, endNeuron) {
        for (let neuron = startNeuron; neuron < endNeuron; neuron++) {
            for (let i = 0; i < this.neuronData.xValues.length; i++) {
                const hiddenValue = this.neuronData.hiddenActivations[i * this.neuronData.numNeurons + neuron];
                const weight = this.neuronData.weights[neuron];
                cumulativeContributions[i] += hiddenValue * weight;
            }
        }

        this.updateChartWithApproximation(cumulativeContributions);
    }

    highlightNeuronGroup(startNeuron, endNeuron) {
        this.drawNetwork(startNeuron, endNeuron);
    }

    updateChartWithApproximation(cumulativeContributions) {
        const datasets = [
            {
                label: 'Target Function',
                data: Array.from(this.neuronData.xValues).map((x, i) => ({
                    x, 
                    y: this.neuronData.yTarget[i]
                })),
                borderColor: '#ef4444',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            },
            {
                label: 'Neural Network Approximation',
                data: Array.from(this.neuronData.xValues).map((x, i) => ({
                    x, 
                    y: cumulativeContributions[i]
                })),
                borderColor: '#06b6d4',
                borderWidth: 3,
                pointRadius: 0,
                fill: false
            }
        ];
        
        this.chart.data.datasets = datasets;
        this.chart.update('none');
    }

    showFinalResult() {
        this.drawNetwork();
        
        const datasets = [
            {
                label: 'Target Function',
                data: Array.from(this.neuronData.xValues).map((x, i) => ({
                    x, 
                    y: this.neuronData.yTarget[i]
                })),
                borderColor: '#ef4444',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            },
            {
                label: 'Neural Network Output',
                data: Array.from(this.neuronData.xValues).map((x, i) => ({
                    x, 
                    y: this.neuronData.predictions[i]
                })),
                borderColor: '#06b6d4',
                borderWidth: 3,
                pointRadius: 0,
                fill: false
            }
        ];
                
        const colors = ['#22c55e', '#a855f7', '#f59e0b', '#ec4899',
        '#14b8a6', '#f97316', '#8b5cf6', '#06b6d4'];
        const maxNeuronsToShow = this.neuronData.numNeurons;

            for (let neuron = 0; neuron < maxNeuronsToShow; neuron++) {
                const contributions = [];
                for (let i = 0; i < this.neuronData.xValues.length; i++) {
                    const hiddenValue = this.neuronData.hiddenActivations[i
        * this.neuronData.numNeurons + neuron];
                    const weight = this.neuronData.weights[neuron];
                    contributions.push({
                        x: this.neuronData.xValues[i],
                        y: hiddenValue * weight
                    });
                }

                datasets.push({
                    data: contributions,
                    borderColor: colors[neuron % colors.length],
                    borderWidth: 0.3,
                    pointRadius: 0,
                    fill: false,
                    showLine: true,
                    pointStyle: false
                });
            }

            this.chart.data.datasets = datasets;
            this.chart.update('active');
            
            // Show custom legend
            document.getElementById('custom-legend').style.display = 'flex';
    }

    updateChartZoom() {
        const baseRange = 10; // Base range from -5 to 5
        const range = baseRange / this.chartZoom;
        const centerX = this.chartPanX;
        const centerY = this.chartPanY;
        
        this.chart.options.scales.x.min = centerX - range/2;
        this.chart.options.scales.x.max = centerX + range/2;
        this.chart.options.scales.y.min = centerY - range/2;
        this.chart.options.scales.y.max = centerY + range/2;
        
        this.chart.update('none');
    }

    setupChartZoom() {
        const chartCanvas = document.getElementById('main-chart');
        
        // Mouse wheel zoom
        chartCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.chartZoom = Math.min(Math.max(0.2, this.chartZoom * delta), 5);
            this.updateChartZoom();
        });

        // Mouse drag to pan
        let isDragging = false;
        let lastX, lastY;

        chartCanvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.offsetX;
            lastY = e.offsetY;
        });

        chartCanvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = (e.offsetX - lastX) * 0.01 / this.chartZoom;
                const deltaY = (e.offsetY - lastY) * 0.01 / this.chartZoom;
                this.chartPanX -= deltaX;
                this.chartPanY += deltaY; // Inverted for chart coordinates
                lastX = e.offsetX;
                lastY = e.offsetY;
                this.updateChartZoom();
            }
        });

        chartCanvas.addEventListener('mouseup', () => {
            isDragging = false;
        });

        chartCanvas.addEventListener('mouseleave', () => {
            isDragging = false;
        });
    }

}

document.addEventListener('DOMContentLoaded', () => {
    new UniversalApproximator();
});
