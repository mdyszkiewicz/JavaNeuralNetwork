package mattd;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jspecify.annotations.NonNull;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuronTest {
    @Test
    void testWeights() {
        // given
        List<Pair<Neuron, Double>> inputs = Arrays.asList(
                createInputNeuron(0.0, 0.4),
                createInputNeuron(1.0, 0.2));
        Neuron neuron = Neuron.NeuronBuilder.createNeuronWithInputs(inputs, 0.5);

        // when
        List<Double> weights = neuron.getWeights();

        // then
        assertEquals(2, weights.size());
        assertTrue(weights.containsAll(Arrays.asList(0.4, 0.2)));
    }

    @Test
    void testValue() {
        // given
        List<Pair<Neuron, Double>> inputs = Arrays.asList(
                createInputNeuron(0.0, 0.4),
                createInputNeuron(1.0, 0.2));
        Neuron neuron = Neuron.NeuronBuilder.createNeuronWithInputsUsingOutputTransformingFunction(inputs, 0.5, new Neuron.TransparentValue());

        // when
        Double value = neuron.getValue();

        // then
        assertEquals(0.7, value);
    }

    private static @NonNull ImmutablePair<Neuron, Double> createInputNeuron(double inputValue, double weight) {
        return new ImmutablePair<>(Neuron.NeuronBuilder
                .createNeuronWithValueUsingOutputTransformingFunction(inputValue, new Neuron.TransparentValue()), weight);
    }
}