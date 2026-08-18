package mattd;

import com.google.common.base.Function;
import com.google.common.collect.Maps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<List<Double>>> weights;
    private final Map<Integer, List<Neuron>> layeredNeurons;


    public NeuralNetwork(List<Integer> layerSizes) {
        vetoNotEnoughLayers(layerSizes);
        weights = new HashMap<>();
        layeredNeurons = new HashMap<>();
        for (int layerNumber = 1; layerNumber < layerSizes.size(); layerNumber++) {
            int previousLayerNumber = layerNumber - 1;
            weights.put(layerNumber, IntStream.range(0, layerSizes.get(layerNumber))
                    .mapToObj(nodeNumber -> IntStream.range(0, layerSizes.get(previousLayerNumber))
                            .mapToObj(prevLayerConnection -> Math.random()).collect(Collectors.toList()))
                    .collect(Collectors.toList()));
            if (layerNumber == 1) {
                layeredNeurons.put(layerNumber, createNeurons(
                        layerSizes.get(layerNumber), () -> Neuron.NeuronBuilder.createNeuronWithValue(Math.random())));
            } else {
                layeredNeurons.put(layerNumber, createNeurons(
                        layerSizes.get(layerNumber), () -> Neuron.NeuronBuilder.createNeuronWithValue(Math.random())));
            }
        }

    }

    private static List<Neuron> createNeurons(Integer neuronCount, Supplier<Neuron> neuronCreator) {
        return IntStream.range(0, neuronCount)
                .mapToObj(nodeNUmber -> {
                    return neuronCreator.get();
                })
                .collect(Collectors.toList());
    }

    public Map<Integer, List<List<Double>>> getWeights() {
        return weights;
    }

    private void vetoNotEnoughLayers(List<Integer> layerSizes) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new RuntimeException("Layer sizes must be greater than 1 (and non null)");
        }
    }

    public Map<Integer, List<Double>> getBiases() {
        return Maps.transformValues(layeredNeurons, neurons -> neurons.stream()
                .map((Function<Neuron, Double>) Neuron::getBias).collect(Collectors.toList()));
    }
}
