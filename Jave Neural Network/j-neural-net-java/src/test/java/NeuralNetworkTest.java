import org.junit.Test;


import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class NeuralNetworkTest {

    private static void assertBiasesAreMostlyNonZero(NeuralNetwork neuralNetwork) {
        long nonZeroBiases = 0;
        for (List<Double> biases : neuralNetwork.getBiases().values()){
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

        assertEquals(4, neuralNetwork.getBiases().get(1).size());
        assertBiasesAreMostlyNonZero(neuralNetwork);
    }

    @Test(expected = RuntimeException.class)
    public void getInvalidNetworkSize() {
        new NeuralNetwork(Arrays.asList(16));
    }

    @Test(expected = RuntimeException.class)
    public void getNullNetworkSize() {
        new NeuralNetwork(null);
    }

}