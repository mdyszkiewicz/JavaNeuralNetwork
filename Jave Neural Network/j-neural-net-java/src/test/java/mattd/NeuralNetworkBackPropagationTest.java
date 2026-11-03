package mattd;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NeuralNetworkBackPropagationTest {

    @Test
    void simplestNotGateLearning() {
        /*
            Here we try the SIMPLEST example - one neuron supposed to flip the input.
            so the structure is the easiest: INPUT -> NEURON -> OUTPUT

        * */
        // given
        Double[][] inputs = {{0.0}, {1.0}};
        Double[][] expected = {{1.0}, {0.0}};
        NeuralNetwork neuralNetwork = new NeuralNetwork(Arrays.asList(1, 1));

        // when
        neuralNetwork.train(inputs, expected, 50, 0.1);

        // then
        neuralNetwork.setInput(Collections.singletonList(1.0));
        assertEquals(0, Math.round(neuralNetwork.forwardPropagation().get(0)));
        neuralNetwork.setInput(Collections.singletonList(0.0));
        assertEquals(1, Math.round(neuralNetwork.forwardPropagation().get(0)));

    }
}