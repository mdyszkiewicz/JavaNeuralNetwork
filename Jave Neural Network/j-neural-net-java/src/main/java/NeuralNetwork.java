import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<Double>> biases;

    public NeuralNetwork(List<Integer> layerSizes) {
        vetoNotEnoughLayers(layerSizes);
        biases = IntStream.range(1, layerSizes.size()).boxed().collect(Collectors.toMap(
                layerNumber -> layerNumber,
                layerNumber -> IntStream.range(0, layerSizes.get(layerNumber))
                        .mapToObj(i -> Math.random()).collect(Collectors.toList())));
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
