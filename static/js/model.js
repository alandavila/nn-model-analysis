

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 1;//10

let mobilenet;
const mobilenetDemo = async () => {
  //status('Loading model...');// for   <div id="status"></div>

  mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  //status('');// for   <div id="status"></div>

};

var prediction_data = {predictions : []};
var grouped_by_class = {};
/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  //status('Predicting...');

  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime = performance.now() - startTime;
  //status(`Done in ${Math.floor(totalTime)}ms`);

  // append data to prediction_date which will be our
  //souce for plots
  appendToData(classes);

}

function appendToData(classes){
  //console.log(classes);
  prediction_data.predictions.push({
    class: classes[0].className,
    probability: classes[0].probability.toFixed(3)
  });

}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
 async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    prediction_data = {predictions : []};
    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
  console.log(prediction_data.predictions)
});

function group(list){
  var out_obj = {};
  list.forEach( function(item){
    var key  =  item.class;
    out_obj[key] = out_obj[key] || [];
    out_obj[key].push(item);
  });
  return out_obj;
}

function getFrequencies(grouped){
  Object.keys(grouped).forEach(function(key) {
    //console.log(key, grouped[key]);
    class_frequency.push([key, grouped[key].length]);
  });
}

var class_frequency = [];
var top_ten_freq = [];
var top_ten_class_probas = {};

function getTopTenPredictions(){
  /*
  group prediction_data.predictions into predicted classes,
  get the top ten most frequet (top_ten_freq) and
  get the probabilities assigned to each image that belongs
  to those top ten classes for each class (top_ten_class_probas)
  */
  // group predictions by common class
  grouped_by_class = {}
  class_frequency = []
  grouped_by_class =  group( prediction_data.predictions) ;
  // sort classes by frequency
  getFrequencies(grouped_by_class);
  class_frequency.sort(function(a,b){
    if (a[1] > b[1]) return -1;
    if (a[1] < b[1]) return 1;
    return 0;
  });
  //generate array of top 10 classes' frequency
  for(var i = 0; i < 10; i++){
    top_ten_freq.push(class_frequency[i]);
  }
  //generate array of top 10 classes' array of probabilities
  top_ten_freq.forEach(function(item){
    var current_class = item[0];
    top_ten_class_probas[current_class] = top_ten_class_probas[current_class] || [];
    grouped_by_class[current_class].forEach(function(item){
      top_ten_class_probas[current_class].push(
        parseFloat(item.probability)
      );
    });
  });
}

var color_pallete = [
  '#FA0410',
  '#FA046C',
  '#DA12AF',
  '#D56ED5',
  '#DBCDE1',
  '#6F30CA',
  '#46379C',
  '#1E38DC',
  '#53ABEC',
  '#12E2EC'
]

function getProbasArray(){
  var probas_array = [];
  for(var i = 0; i < 10; i++){
    probas_array.push({
      type: 'violin',
      y: top_ten_class_probas[top_ten_freq[i][0]],//unpack(rows, 'total_bill'),
      x: top_ten_freq[i][0],
      name: top_ten_freq[i][0],
      points: 'none',
      box: {
        visible: true
      },
      boxpoints: false,
      line: {
        color: 'black'
      },
      fillcolor: color_pallete[i],
      opacity: 0.6,
      meanline: {
        visible: true
      },
      //x0: top_ten_freq[i][0]
    })
  }
  return probas_array;
}

function makeViolinPlots(){

  var data = getProbasArray();
  var layout = {
    title: "Class probability dist. for top 10 classes",
    height: 600,
    width: 600,
    showlegend: false,
    yaxis: {
      zeroline: false
    }
  }

  Plotly.plot('violinplot', data, layout);
}

function makePlots(){
  getTopTenPredictions();
  //TODO histogram plot via d3
  //TODO boxplot via d3
  makeViolinPlots();
  //TODO cluster plot via d

}

mobilenetDemo();
