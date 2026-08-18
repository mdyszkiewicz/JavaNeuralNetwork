package mattd;

import com.google.common.base.Function;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jspecify.annotations.NonNull;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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

    private static Integer getDesiredSizeOfLayer(int layerNo, List<Integer> layerSizes) {
        return layerSizes.get(layerNo - 1); // size of layer is a list - it's index is @-1
    }

    private static List<Neuron> createNeurons(Integer neuronCount, Supplier<Neuron> neuronCreator) {
        return IntStream.range(0, neuronCount)
                .mapToObj(nodeNUmber -> neuronCreator.get())
                .collect(Collectors.toList());
    }

    public Map<Integer, List<List<Double>>> getWeights() {
        return Maps.transformValues(allButTheFirstLayer(), neurons -> Objects.requireNonNull(neurons).stream()
                .map((Function<Neuron, List<Double>>) Neuron::getWeights).collect(Collectors.toList()));
    }

    private void vetoNotEnoughLayers(List<Integer> layerSizes) {
        if (layerSizes == null || layerSizes.size() < 2) {
            throw new RuntimeException("Layer sizes must be greater than 1 (and non null)");
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
}
