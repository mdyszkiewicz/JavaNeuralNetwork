package mattd;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class NeuralNetworkTest {

    @Test
    public void getBiasesAndWeights() {
        // given
        NeuralNetwork neuralNetwork = new NeuralNetwork(Arrays.asList(16, 4));

        // when
        Map<Integer, List<Double>> biases = neuralNetwork.getBiases();
        Map<Integer, List<List<Double>>> weightMap = neuralNetwork.getWeights();

        // then
        assertEquals(4, biases.get(2).size());
        assertBiasesAreMostlyNonZero(biases);
        assertEquals(1, weightMap.size());
        weightMap.get(2).forEach(weights -> assertEquals(16, weights.size()));
    }

    @Test
    public void testForwardPropagation() {
        // given
        NeuralNetwork neuralNetwork = new NeuralNetwork(Arrays.asList(2, 1, 1));
        List<Double> inputs = Arrays.asList(0.0, 1.0);
        neuralNetwork.setInput(inputs);
        Double biasLayer2 = neuralNetwork.getBiases().get(2).get(0);
        List<Double> weightsLayer2 = neuralNetwork.getWeights().get(2).get(0);
        Double biasLayer3 = neuralNetwork.getBiases().get(3).get(0);
        List<Double> weightsLayer3 = neuralNetwork.getWeights().get(3).get(0);

        // when
        List<Double> networkOutputs = neuralNetwork.forwardPropagation();

        // then
        // in this case the network output is a single number (number of neurons in last layer is 1)
        assertEquals(1, networkOutputs.size());
        // the value should be:
        // sigmoidFunction( input1 * weight1_layer2 + input2 * weight2_layer2 + biasLayer2 ) = resultLayer2
        double result = inputs.get(0) * weightsLayer2.get(0) + inputs.get(1) * weightsLayer2.get(1) + biasLayer2;
        Neuron.SigmoidFunction sigmoidFunction = new Neuron.SigmoidFunction();
        result = sigmoidFunction.apply(result);
        // sigmoidFunction( resultLayer2 * weightLayer3 + biasLayer2 ) = resultLayer3 = output
        result = sigmoidFunction.apply(result * weightsLayer3.get(0) + biasLayer3);
        assertEquals(result, networkOutputs.get(0));
    }


    @Test
    public void getInvalidNetworkSize() {
        assertThrows(RuntimeException.class, () -> new NeuralNetwork(null));
        assertThrows(RuntimeException.class, () -> new NeuralNetwork(Collections.singletonList(16)));
    }

    private void assertBiasesAreMostlyNonZero(Map<Integer, List<Double>> biasMap) {
        long nonZeroBiases = 0;
        for (List<Double> biases : biasMap.values()) {
            for (Double bias : biases)
                if (0 != Double.compare(bias, 0.0d)) nonZeroBiases++;
        }
        assertEquals(1, biasMap.size());
        assertNotEquals(0, nonZeroBiases);
    }
}