package mattd;


import org.apache.commons.lang3.tuple.Pair;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class Neuron {
    private List<Pair<Neuron, Double>> weightedInputs;
    private Double bias;

    Neuron(List<Pair<Neuron, Double>> weightedInputs, Double bias) {
        this.weightedInputs = weightedInputs;
        this.bias = bias;
    }

    public Double getValue() {
        return 0.0;
    }

    public Double getBias() {
        return bias;
    }

    public List<Double> getWeights() {
        return weightedInputs.stream().map(Pair::getValue).collect(Collectors.toList());
    }

    public static class NeuronBuilder {

        public static Neuron createNeuronWithInputs(List<Pair<Neuron, Double>> weightedInputs, Double bias) {
            return new Neuron(weightedInputs, bias);
        }

        public static Neuron createNeuronWithValue(Double value) {
            return new Neuron(Collections.emptyList(), value);
        }
    }
}
