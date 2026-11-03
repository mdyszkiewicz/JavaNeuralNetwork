package mattd;

import com.google.common.base.Function;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jspecify.annotations.NonNull;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<Neuron>> layeredNeurons;

    public NeuralNetwork(List<Integer> layerSizes) {
        vetoNotEnoughLayers(layerSizes);
        layeredNeurons = new HashMap<>();
        // initial ("input") neurons
        layeredNeurons.put(1, createNeurons(
                getDesiredSizeOfLayer(1, layerSizes), () -> Neuron.NeuronBuilder.createNeuronWithValue(Math.random())));

        for (int layerNumber = 2; layerNumber <= layerSizes.size(); layerNumber++) {
            int prevLayer = layerNumber - 1;
            layeredNeurons.put(layerNumber, createNeurons(
                    getDesiredSizeOfLayer(layerNumber, layerSizes),
                    () -> Neuron.NeuronBuilder.createNeuronWithInputs(layeredNeurons.get(prevLayer).stream().map(
                                    (Function<Neuron, Pair<Neuron, Double>>)
                                            previousLeyerNeuron -> new MutablePair<>(previousLeyerNeuron, Math.random()))
                            .collect(Collectors.toList()), Math.random())));
        }

    }

    private Integer getDesiredSizeOfLayer(int layerNo, List<Integer> layerSizes) {
        return layerSizes.get(layerNo - 1); // size of layer is a list - it's index is @-1
    }

    private List<Neuron> createNeurons(Integer neuronCount, Supplier<Neuron> neuronCreator) {
        return IntStream.range(0, neuronCount)
                .mapToObj(nodeNUmber -> neuronCreator.get())
                .collect(Collectors.toList());
    }

    /**
     * @return Returns weights of layers. Layers are numbered from 1.
     */
    public Map<Integer, List<List<Double>>> getWeights() {
        return Maps.transformValues(allButTheFirstLayer(), neurons -> Objects.requireNonNull(neurons).stream()
                .map((Function<Neuron, List<Double>>) Neuron::getWeights).collect(Collectors.toList()));
    }

    private void vetoNotEnoughLayers(List<Integer> layerSizes) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new RuntimeException("Layer sizes must be greater than 1 (and non null)");
        }
    }

    public void setInput(List<Double> list) {
        List<Neuron> neuronsLayer1 = layeredNeurons.get(1);
        if (list.size() != neuronsLayer1.size()) {
            throw new RuntimeException(
                    String.format("Input size different than neurons in the 1st layer count%d / %d", list.size(), neuronsLayer1.size()));
        }
        for (int i = 0; i < list.size(); i++) {
            neuronsLayer1.get(i).setBias(list.get(i));
        }
    }

    public Map<Integer, List<Double>> getBiases() {

        return Maps.transformValues(allButTheFirstLayer(),
                neurons -> Objects.requireNonNull(neurons).stream().map(
                        (Function<Neuron, Double>) Neuron::getBias).collect(Collectors.toList()));
    }

    private @NonNull Map<Integer, List<Neuron>> allButTheFirstLayer() {
        return Maps.filterKeys(layeredNeurons, layerNumber -> layerNumber != null && layerNumber != 1);
    }

    public List<Double> forwardPropagation() {
        return layeredNeurons.get(layeredNeurons.size()).stream().map(Neuron::getValue).collect(Collectors.toList());
    }

    public void train(Double[][] inputs, Double[][] expectedOutputs, int iterations, Double learningRate) {
        for (int iter = 1; iter <= iterations; iter++) {
            logDebug("Iteration: ", String.valueOf(iter));
            // 1. choose input and expected output
            Double[] input = inputs[iter % inputs.length];
            Double[] expectedOutput = expectedOutputs[iter % expectedOutputs.length];
            // 2. Calculate network output and error
            setInput(Arrays.asList(input));

            List<Double> results = new ArrayList<>();
            List<Neuron> lastLayer = layeredNeurons.get(layeredNeurons.size());
            double error = 0.0;
            for (int neuronNumber = 0; neuronNumber < lastLayer.size(); neuronNumber++) {
                Double difference = expectedOutput[neuronNumber] - lastLayer.get(neuronNumber).getValue();
                error = error + (difference * difference);
                results.add(lastLayer.get(neuronNumber).getValue());
            }
            logDebug("Calculated output: ", results.stream().map(Object::toString).collect(Collectors.joining(", ")));
            logDebug("Sum of squared errors: ", Double.toString(error));
            // 3. calculate gradient
            List<Double> gradient;
            for (int neuronNumber = 0; neuronNumber < lastLayer.size(); neuronNumber++) {
                Neuron outputNeuron = lastLayer.get(neuronNumber);
                // calculate dc/do = 2(o - expectedOutput)
                Double gradientOfCostFunctionVsOutput = 2 * (outputNeuron.getValue() - expectedOutput[neuronNumber]);
                // TODO: - here we need to add the derivative with respect to the transformation function!

                // calculate do/dw for each weight (it's simply input form previous layer) (o = i*w + b) so do/dw = i
                List<Double> valuesFromPreviousLayer = layeredNeurons.get(layeredNeurons.size() - 1).stream().map(Neuron::getValue).collect(Collectors.toList());
                // do/db (bias is a constant so it's derivative is always 1)
                Double biasDerivative = 1.0;
                gradient = valuesFromPreviousLayer.stream().map(value -> value * gradientOfCostFunctionVsOutput).collect(Collectors.toList());
                gradient.add(biasDerivative * gradientOfCostFunctionVsOutput);
                logDebug("Calculated gradient: ", gradient.stream().map(Object::toString).collect(Collectors.joining(", ")));

                outputNeuron.applyGradient(gradient, learningRate);
            }


        }
    }


    private void logDebug(String prefix, String variable) {
        System.out.println(prefix + variable);
    }
}
