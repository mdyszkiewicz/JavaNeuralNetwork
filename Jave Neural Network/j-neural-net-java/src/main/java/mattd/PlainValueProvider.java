package mattd;

public class PlainValueProvider implements ValueProvider {

    private double value;

    public PlainValueProvider(double value) {
        this.value = value;
    }

    @Override
    public Double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
