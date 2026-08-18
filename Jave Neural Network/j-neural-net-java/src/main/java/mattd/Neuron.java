package mattd;


import org.apache.commons.lang3.tuple.Pair;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Neuron {
    private final List<Pair<Neuron, Double>> weightedInputs;
    private final Double bias;
    private final TransformationFunction transformationFunction;

    Neuron(List<Pair<Neuron, Double>> weightedInputs, Double bias, TransformationFunction transformationFunction) {
        this.weightedInputs = weightedInputs;
        this.bias = bias;
        this.transformationFunction = transformationFunction; // no other implementations yet
    }


    public Double getValue() {
        return transformationFunction.apply(weightedInputs.stream()
                .mapToDouble(weightedNeuron -> weightedNeuron.getRight() * weightedNeuron.getLeft().getValue())
                .reduce(Double::sum).orElse(0.0) + getBias());
    }

    public Double getBias() {
        return bias;
    }

    public List<Double> getWeights() {
        return weightedInputs.stream().map(Pair::getValue).collect(Collectors.toList());
    }

    public static class NeuronBuilder {

        public static final SigmoidFunction DEFAULT_TRANSFORMATION_FUNCTION = new SigmoidFunction();

        public static Neuron createNeuronWithInputs(List<Pair<Neuron, Double>> weightedInputs, Double bias) {
            return createNeuronWithInputsUsingOutputTransformingFunction(weightedInputs, bias, DEFAULT_TRANSFORMATION_FUNCTION);
        }

        public static Neuron createNeuronWithInputsUsingOutputTransformingFunction(List<Pair<Neuron, Double>> weightedInputs, Double bias, TransformationFunction transformationFunction) {
            return new Neuron(weightedInputs, bias, transformationFunction);
        }

        public static Neuron createNeuronWithValue(Double value) {
            return createNeuronWithValueUsingOutputTransformingFunction(value, DEFAULT_TRANSFORMATION_FUNCTION);
        }

        public static Neuron createNeuronWithValueUsingOutputTransformingFunction(Double value, TransformationFunction defaultTransformationFunction) {
            return createNeuronWithInputsUsingOutputTransformingFunction(Collections.emptyList(), value, defaultTransformationFunction);
        }

    }


    public interface TransformationFunction extends Function<Double, Double> {

    }

    public static class SigmoidFunction implements TransformationFunction {
        @Override
        public Double apply(Double value) {
            return 1 / (1 + Math.exp(-value));
        }
    }


    public static class TransparentValue implements TransformationFunction {
        @Override
        public Double apply(Double value) {
            return value;
        }
    }
}
