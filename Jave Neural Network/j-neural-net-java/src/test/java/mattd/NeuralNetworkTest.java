package mattd;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class NeuralNetworkTest {

    private static void assertBiasesAreMostlyNonZero(NeuralNetwork neuralNetwork) {
        long nonZeroBiases = 0;
        for (List<Double> biases : neuralNetwork.getBiases().values()) {
            for (Double bias : biases) {
                if (0 != Double.compare(bias, 0.0d)) {
                    nonZeroBiases++;
                }
            }
        }
        assertEquals(1, neuralNetwork.getBiases().size());
        assertNotEquals(0, nonZeroBiases);
    }

    @Test
    public void getBiases() {
        NeuralNetwork neuralNetwork = new NeuralNetwork(Arrays.asList(16, 4));

        assertEquals(4, neuralNetwork.getBiases().get(2).size());
        assertBiasesAreMostlyNonZero(neuralNetwork);
        assertEquals(1, neuralNetwork.getWeights().size());

        neuralNetwork.getWeights().get(2).forEach(weights -> assertEquals(16, weights.size()));
    }

    @Test
    public void getInvalidNetworkSize() {
        assertThrows(RuntimeException.class, () -> new NeuralNetwork(null));
        assertThrows(RuntimeException.class, () -> new NeuralNetwork(Collections.singletonList(16)));
    }

}