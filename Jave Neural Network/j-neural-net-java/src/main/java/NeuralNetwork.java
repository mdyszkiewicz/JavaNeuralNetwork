import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<Double>> biases;
    private final Map<Integer, List<List<Double>>> weights;

    public NeuralNetwork(List<Integer> layerSizes) {
        vetoNotEnoughLayers(layerSizes);
        biases = new HashMap<>();
        weights = new HashMap<>();
        for (int layerNumber = 1; layerNumber < layerSizes.size(); layerNumber++) {
            biases.put(layerNumber, IntStream.range(0, layerSizes.get(layerNumber))
                    .mapToObj(nodeNUmber -> Math.random()).collect(Collectors.toList()));
            int previousLayerNumber = layerNumber - 1;
            weights.put(layerNumber, IntStream.range(0, layerSizes.get(layerNumber))
                    .mapToObj(nodeNumber -> IntStream.range(0, layerSizes.get(previousLayerNumber))
                            .mapToObj(prevLayerConnection -> Math.random()).collect(Collectors.toList()))
                    .collect(Collectors.toList()));
        }
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
        return biases;
    }
}
