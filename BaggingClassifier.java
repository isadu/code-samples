package ml.classifiers;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;
import ml.utils.HashMapCounterDouble;

import java.util.ArrayList;
import java.util.Map;

public class BaggingClassifier implements Classifier{
    final private ArrayList<Classifier> classifiers;
    private int type = 0;
    private int m = 10;
    private double sampleProportion = 0.5;
    private ClassifierFactory factory;
    private boolean useConfidence = false;

    public BaggingClassifier() {
        classifiers = new ArrayList<>();
    }

    public BaggingClassifier(int type, int m, double sampleProportion) {
        classifiers = new ArrayList<>();
        this.type = type;
        this.m = m;
        this.sampleProportion = sampleProportion;
        this.factory = new ClassifierFactory(type);
    }

    /**
     * Train m classifiers on data sets that are sampleProportion of the original data set.
     *
     * @param data training data
     */
    @Override
    public void train(DataSet data) {
        //DataSetSplit newDataSet;
        for (int i=0; i<m; i++) {
            final DataSetSplit newDataSet = data.split(sampleProportion);
            final Classifier newClassifier = factory.getClassifier();
            newClassifier.train(newDataSet.getTrain());
            classifiers.add(newClassifier);
        }
    }

    /**
     * Classify the example using the label most often predicted by the subclassifiers.
     *
     * @param example example to predict
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        if (useConfidence) return classifyUsingConfidence(example);
        final HashMapCounter<Double> scoreForLabel = new HashMapCounter<>();
        for (Classifier c : classifiers) {
            final double label = c.classify(example);
            scoreForLabel.increment(label);
        }
        final ArrayList<Map.Entry<Double, Integer>> set = scoreForLabel.sortedEntrySet();
        return set.get(0).getKey();
    }

    /**
     * Classify using weighted votes based on confidence of the subclassifiers' predictions.
     *
     * @param example example to classify
     * @return predicted label
     */
    public double classifyUsingConfidence(Example example) {
        final HashMapCounterDouble<Double> scoreForLabel = new HashMapCounterDouble<>();
        for (Classifier c : this.classifiers) {
            final double label = c.classify(example);
            final double confidence = c.confidence(example);
            scoreForLabel.increment(label, confidence);
        }
        final ArrayList<Map.Entry<Double, Double>> set = scoreForLabel.sortedEntrySet();
        return set.get(0).getKey();
    }

    /**
     * Returns the sum of the subclassifiers' confidence for the selected label.
     *
     * @param example example to get confidence for
     * @return confidence sum
     */
    @Override
    public double confidence(Example example) {
        final HashMapCounterDouble<Double> scoreForLabel = new HashMapCounterDouble<>();
        for (Classifier c : this.classifiers) {
            final double label = c.classify(example);
            final double confidence = c.confidence(example);
            scoreForLabel.increment(label, confidence);
        }
        final ArrayList<Map.Entry<Double, Double>> set = scoreForLabel.sortedEntrySet();
        return set.get(0).getValue();
    }

    /**
     * Set the type of classifiers to use.
     *
     * @param type classifier type
     */
    public void setType(int type) {
        this.type = type;
    }

    /**
     * Set the number of classifiers to use
     *
     * @param m number of classifiers
     */
    public void setM(int m) {
        this.m = m;
    }

    /**
     * Set the proportion of the original dataset to use for each bootstrap data sample.
     *
     * @param sampleProportion proportion to use
     */
    public void setSampleProportion(double sampleProportion) {
        this.sampleProportion = sampleProportion;
    }

    /**
     * Set using confidence in classification.
     *
     * @param useConfidence whether to use confidence
     */
    public void setUseConfidence(boolean useConfidence) { this.useConfidence = useConfidence; }

    /**
     * Set all the hyperparameters at once.
     *
     * @param type type of subclassifier
     * @param m number of subclassifiers
     * @param prop proportion of original dataset to sample
     */
    public void setHyperparameters(int type, int m, double prop) {
        setType(type);
        setM(m);
        setSampleProportion(prop);
    }

    /**
     * Set the hyperparameters for the subclassifiers this classifier uses.
     * Formats:
     * Decision tree: [depth limit]
     * Gradient descent: [loss, regularization, lambda*100, eta*100, iterations]
     * k-NN: [k]
     * Perceptron: [iterations]
     * Two-layer NN: [eta*100, iterations]
     *
     * @param params parameters for the subclassifiers, as described above
     */
    public void setSubclassifierHyperparameters(int[] params) {
        this.factory = new ClassifierFactory(type, params);
    }

    /**
     * Return a classifier of the same type and with the same hyperparameters as the subclassifiers here.
     *
     * @return new classifier
     */
    public Classifier getSingleClassifier() {
        return factory.getClassifier();
    }
}
