const tf = require('@tensorflow/tfjs');
const fetch = require('node-fetch');

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const inputs = data.map(d => d.events);
        const labels = data.map(d => d.participants);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = inputTensor.max();
        const labelMin = inputTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    });
}

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
        x: d.events, y: d.participants,
    }));


    for (let i = 0; i < predictedPoints.length; i++) {
        if (predictedPoints[i] && originalPoints[i]) {
            let diff = {
                xDiff: predictedPoints[i]["x"] / originalPoints[i]["x"] * 100,
                yDiff: predictedPoints[i]["y"] / originalPoints[i]["y"] * 100
            };

            console.log(predictedPoints[i], originalPoints[i], diff);
        }
    }
}

async function run() {
    const body = {
        query: `query TournamentsByCountry($perPage: Int, $cCode: String!) {
      tournaments(query: {
        perPage: $perPage
        filter: {
          countryCode: $cCode
        }
      }) {
        nodes {
          id
          name
          events {
            id
          }
          countryCode
          participants(query: {
            perPage: $perPage
          }) {
            nodes {
              id
              gamerTag
            }
          }
        }
      }
    }
    `,
        variables: JSON.parse("{\n  \"perPage\": 50,\n  \"cCode\": \"US\"\n}")
    };

    let data = [];

    const response = await fetch("https://api.smash.gg/gql/alpha", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer 649706e3a7cd1d79fe9ba1026ad6b9be"
        },
        body: JSON.stringify(body)
    });
    const res = await response.json();
    if ((res.status !== 200) && res.status !== 201) {
        console.log(res["data"]["tournaments"]["nodes"][0]);
        res["data"]["tournaments"]["nodes"].forEach(tournament => {
            if (tournament["participants"]["nodes"] && tournament["events"]) {
                data.push({
                    tournament: tournament["name"],
                    participants: tournament["participants"]["nodes"].length,
                    events: tournament["events"].length
                });
            }
        });
        console.log(data);
        return data;
    } else {
        console.log(await res.error());
    }
}

const promise = run()
    .then(async (data) => {
        const values = data.map(d => ({
            x: d.events,
            y: d.participants
        }));
        const model = createModel();

        const tensorData = convertToTensor(data);
        const { inputs, labels } = tensorData;

        await trainModel(model, inputs, labels);
        console.log('Done training');
        testModel(model, data, tensorData);
        console.log("Done!");
    }).catch((error) => {
        console.log(error);
    });
