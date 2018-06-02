

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
var class_frequency = [];
var top_ten_freq = [];
var top_ten_class_probas = {};
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
  appendToData(classes, imgElement);

}

function appendToData(classes){
  //console.log(classes);
  prediction_data.predictions.push({
    class: classes[0].className,
    probability: classes[0].probability.toFixed(3),
    image: imgElement
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
    class_frequency = [];
    top_ten_freq = [];
    top_ten_class_probas = {};
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
    height: 500,
    width: 960,
    showlegend: false,
    yaxis: {
      zeroline: false
    }
  };

  Plotly.newPlot('violinplot', data, layout);
};

function makeForcePlot(){

 var width = 960;
 var height = 500;

 var color = d3.scaleOrdinal(d3.schemeCategory20);

 var svg = d3.select("#forcePlot").append("svg")
   .attr("width", width)
   .attr("height", height);

   // values for all forces
forceProperties = {
    center: {
        x: 0.5,
        y: 0.5
    },
    charge: {
        enabled: true,
        strength: -30,
        distanceMin: 1,
        distanceMax: 2000
    },
    collide: {
        enabled: true,
        strength: .7,
        iterations: 1,
        radius: 5
    },
    forceX: {
        enabled: false,
        strength: .1,
        x: .5
    },
    forceY: {
        enabled: false,
        strength: .1,
        y: .5
    },
    link: {
        enabled: true,
        distance: 30,
        iterations: 1
    }
}

 var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().distance(100).strength(0.5).id(function(d) { return d.name; }))
    .force("charge", d3.forceManyBody().strength(-3) ) //or null
    .force("center", d3.forceCenter(width / 2, height / 2))
    //.force("forceX", d3.forceX())
    //.force("forceY", d3.forceY());




 var nodes = [];
 var links = [];

//parent nodes
for (let x = 0; x < 10; x++) {
  nodes.push({"name": top_ten_freq[x][0], "freq": (top_ten_freq[x][1]), "parent": true});
};

//children image nodes
for (let x = 0; x < top_ten_class_probas[top_ten_freq[x][0]].length; x++) {
  for (let i = 0; i < top_ten_class_probas[top_ten_freq[x][0]].length; i++) {
    nodes.push({"name": "Class"+x+" "+"image"+i, "freq": (top_ten_class_probas[top_ten_freq[x][0]][i])*10, "child": true, "src":prediction_data.predictions[i].image.src})
  };
};

//links to parent node
for (let x = 0; x < top_ten_class_probas[top_ten_freq[x][0]].length; x++) {
  for (let j=0; j < top_ten_class_probas[top_ten_freq[x][0]].length; j++){
    links.push({"source": top_ten_freq[x][0], "target": "Class"+x+" "+"image"+j})
  };
};

//links from parent node to parent node
for (let x = 0; x < 9; x++) {
  links.push({"source": top_ten_freq[x][0], "target": (top_ten_freq[x+1][0])});
};

var link = svg.append("g")
        .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
        .attr("stroke-width", 2);

    var node = svg.append("g")
        .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .on("click", handleClick)
      .on("mouseover", handleMouseOver)
      .on("mouseout", handleMouseOut)
        .attr("r", d => (d.freq))
        .attr("fill", colorizer)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));



    node.append("title")
        .text(function(d) { return (d.name +" "+ d.freq) ; });

    simulation
        .nodes(nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(links);

    function colorizer(d) {
      //if parent node, color orange, else default color (ternary operation)
      //return nodes.parent == true ? "orange": "blue";
      return d.parent ? "orange": "rgb(31, 119, 180)";

    };

    // Create Event Handlers for mouse
    function handleClick() {
      //get the image and display either as popup or in a new div
      alert("You clicked me!" + this.node);
    };
    function handleMouseOver() {  // Add interactivity
        // Use D3 to select element, change color and size
          d3.select(this)
            .style("fill", "#00AAFF");
          // Get current event info
          console.log(d3.event);
          // Get x & y co-ordinates
          //console.log(d3.mouse(this));
      };

    function handleMouseOut() {

      d3.select(this)
        .style("fill", colorizer);

    }

    function ticked() {
      link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

      node
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    };
//  });

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  };

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  };

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  };

};

function makeBarGraph() {
var x_data = [];
var y_data = [];

for(i=0; i<10;i++){
  x_data.push(top_ten_freq[i][0]);
  y_data.push(top_ten_freq[i][1]);
}

var trace1 = {
  type: "bar",
//for(var i = 0; i < 10; i++){
  x: x_data,
  y: y_data,
//}
  name: "Top Frequency",
//  orientation: "h",
  fillcolor: color_pallete,
  opacity: 0.6,
};
var data1 = [trace1];
var layout = {
title: "Top Frequency Bar Graph ",
  height: 500,
  width: 960,
xaxis: { title: "Predicted Class" },
yaxis: { title: "Class Frequency" }
};

Plotly.newPlot('bargraph', data1, layout);
}

function makePlots(){
  //clear force plot if already present
  d3.select("#forcePlot > svg").remove();
  getTopTenPredictions();
  // histogram plot via d3
  makeBarGraph();
  // cluster plot via d
  makeForcePlot();
  //boxplot via d3
  makeViolinPlots();

}

mobilenetDemo();
