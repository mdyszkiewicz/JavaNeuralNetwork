import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class NeuralNetwork {

    private final Map<Integer, List<Double>> biases;
    private final List<Integer> layerSizes;

    public NeuralNetwork(List<Integer> layerSizes) {
        this.layerSizes = layerSizes;
        biases = IntStream.range(1, layerSizes.size()).boxed().collect(Collectors.toMap(
                layerNumber -> layerNumber,
                layerNumber -> IntStream.range(0, layerSizes.get(layerNumber))
                        .mapToObj(i -> Math.random()).collect(Collectors.toList())));
    }

    public Map<Integer, List<Double>> getBiases() {
        return biases;
    }
}
