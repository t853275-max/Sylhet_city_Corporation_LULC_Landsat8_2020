// ROI
Map.centerObject(roi, 10);
Map.addLayer(roi, {}, 'ROI');

// Landsat 8 Collection
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");

var image = l8
  .filterBounds(roi)
  .filterDate('2020-11-01', '2020-12-31')
  .filterMetadata('CLOUD_COVER', 'less_than', 10)
  .median()
  .clip(roi);

// Scale factors
function scale(img) {
  var optical = img.select('SR_B.*')
    .multiply(0.0000275)
    .add(-0.2);
  return img.addBands(optical, null, true);
}

image = scale(image);

// Indices
var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
var ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
var mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');

image = image.addBands([ndvi, ndbi, mndwi]);

// Training samples
var samples = water
  .merge(vegetation)
  .merge(bareland)
  .merge(buildup);

// Bands
var bands = [
  'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4',
  'SR_B5', 'SR_B6', 'SR_B7',
  'NDVI', 'NDBI', 'MNDWI'
];

// Training data
var training = image.select(bands).sampleRegions({
  collection: samples,
  properties: ['Class'],
  scale: 30
});

// Train-test split
var withRandom = training.randomColumn('random');

var trainSet = withRandom.filter(ee.Filter.lt('random', 0.8));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.8));

// Random Forest classifier
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 200
}).train({
  features: trainSet,
  classProperty: 'Class',
  inputProperties: bands
});

// Classification
var classified = image.select(bands).classify(classifier);

Map.addLayer(
  classified,
  {min: 0, max: 3, palette: ['blue', 'green', 'yellow', 'red']},
  'LULC'
);

// Accuracy assessment
var validated = testSet.classify(classifier);

var confusionMatrix = validated.errorMatrix('Class', 'classification');

print('Confusion Matrix', confusionMatrix);
print('Overall Accuracy', confusionMatrix.accuracy());
print('Kappa Coefficient', confusionMatrix.kappa());
print('Producer Accuracy', confusionMatrix.producersAccuracy());
print('User Accuracy', confusionMatrix.consumersAccuracy());

var classified = image.select(bands)
  .classify(classifier)
  .clip(roi);

Export.image.toDrive({
  image: classified,
  description: 'LULC_2020_RF',
  folder: 'GEE_LULC',
  fileNamePrefix: 'LULC_2020_RF',
  region: roi,
  scale: 30,
  maxPixels: 1e13
});


var pixelArea = ee.Image.pixelArea();

// function for class area
function getArea(classValue, name) {
  var areaImage = pixelArea.updateMask(classified.eq(classValue));

  var area = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 30,
    maxPixels: 1e13
  });

  print(name + ' (m²):', area);
  print(name + ' (km²):', ee.Number(area.get('area')).divide(1e6));
}

getArea(0, 'Water');
getArea(1, 'Vegetation');
getArea(2, 'Bareland');
getArea(3, 'Built-up');
