package mattd;

import com.google.common.base.Function;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jspecify.annotations.NonNull;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<Neuron>> layeredNeurons;


    public NeuralNetwork(List<Integer> layerSizes) {
        vetoNotEnoughLayers(layerSizes);
        layeredNeurons = new HashMap<>();
        for (int layerNumber = 1; layerNumber <= layerSizes.size(); layerNumber++) {
            if (layerNumber == 1) {
                layeredNeurons.put(layerNumber, createNeurons(
                        layerSizes.get(0), () -> Neuron.NeuronBuilder.createNeuronWithValue(Math.random())));
            } else {
                int prevLayer = layerNumber - 1;
                layeredNeurons.put(layerNumber, createNeurons(
                        layerSizes.get(layerNumber - 1), () -> Neuron.NeuronBuilder.createNeuronWithInputs(
                                layeredNeurons.get(prevLayer).stream().map(
                                                (Function<Neuron, Pair<Neuron, Double>>) previousLeyerNeuron -> new MutablePair<>(previousLeyerNeuron, Math.random()))
                                        .collect(Collectors.toList()), Math.random())));
            }
        }

    }

    private static List<Neuron> createNeurons(Integer neuronCount, Supplier<Neuron> neuronCreator) {
        return IntStream.range(0, neuronCount)
                .mapToObj(nodeNUmber -> neuronCreator.get())
                .collect(Collectors.toList());
    }

    public Map<Integer, List<List<Double>>> getWeights() {
        return Maps.transformValues(allButTheFirstLayer(), neurons -> neurons.stream()
                .map((Function<Neuron, List<Double>>) Neuron::getWeights).collect(Collectors.toList()));
    }

    private void vetoNotEnoughLayers(List<Integer> layerSizes) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new RuntimeException("Layer sizes must be greater than 1 (and non null)");
        }
    }

    public Map<Integer, List<Double>> getBiases() {
        return Maps.transformValues(allButTheFirstLayer(),
                neurons -> neurons.stream().map(
                        (Function<Neuron, Double>) Neuron::getBias).collect(Collectors.toList()));
    }

    private @NonNull Map<Integer, List<Neuron>> allButTheFirstLayer() {
        return Maps.filterKeys(layeredNeurons, layerNumber -> layerNumber != 1);
    }
}
