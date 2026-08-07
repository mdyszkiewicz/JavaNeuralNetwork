import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


public class NeuralNetwork {

    private final List<List<Double>> biases;
    private final List<Integer> layerSizes;

    public NeuralNetwork(List<Integer> layerSizes) {
        this.layerSizes = layerSizes;
        biases = IntStream.range(0, layerSizes.size())
                .mapToObj(i -> new ArrayList<Double>(layerSizes.get(i)))
                .collect(Collectors.toList());
        for (int layerNo = 0; layerNo < layerSizes.size(); layerNo++) {
            if (layerNo>0){
                List<Double> biasesInCurrentLayer = biases.get(layerNo);
                for (int j = 0; j < layerSizes.get(layerNo); j++) {
                    biasesInCurrentLayer.add(Math.random());
                }
            }
        }
    }

    public List<List<Double>> getBiases() {
        return biases;
    }
}
