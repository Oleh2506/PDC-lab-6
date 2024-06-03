import java.util.Random;

public class Matrix {
    private final double[][] matrix;
    private final int rowNum;
    private final int colNum;

    public Matrix(double[][] matrix) {
        this.matrix = matrix;
        this.rowNum = matrix.length;
        this.colNum = matrix[0].length;
    }

    public Matrix(double[] arr1D, int rowNum, int colNum) {
        if (arr1D.length != rowNum * colNum) {
            throw new IllegalArgumentException("Array length does not match matrix dimensions!");
        }

        this.matrix = new double [rowNum][colNum];
        this.rowNum = rowNum;
        this.colNum = colNum;

        int index = 0;
        for (int i = 0; i < this.rowNum; i++) {
            for (int j = 0; j < this.colNum; j++) {
                this.matrix[i][j] = arr1D[index];
                index++;
            }
        }
    }

    public Matrix(int rowNum, int colNum) {
        this.matrix = new double [rowNum][colNum];
        this.rowNum = rowNum;
        this.colNum = colNum;
    }

    public int getRowCount() { return this.rowNum; }

    public int getColumnCount() { return this.colNum; }

    public double getElement(int i, int j) { return this.matrix[i][j]; }

    public void setElement(int i, int j, double value) { this.matrix[i][j] = value; }

    public void fill(double value) {
        for (int i = 0; i < this.rowNum; i++) {
            for (int j = 0; j < this.colNum; j++) {
                matrix[i][j] = value;
            }
        }
    }

    public void fillRandom() {
        int minValue = 1;
        int maxValue = 10;
        Random rand = new Random();

        for (int i = 0; i < this.rowNum; i++) {
            for (int j = 0; j < this.colNum; j++) {
                matrix[i][j] = rand.nextInt((maxValue - minValue) + 1) + minValue;
            }
        }
    }

    public void setStripeByArr1D(double[] arr1D, int startRow) {
        if (arr1D.length % rowNum != 0 || arr1D.length % colNum != 0) {
            throw new IllegalArgumentException("Array length does not match matrix dimensions!");
        }

        int endRow = arr1D.length / colNum + startRow;
        int index = 0;
        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < colNum; j++) {
                this.matrix[i][j] = arr1D[index];
                index++;
            }
        }
    }

    public void print() {
        for (double[] row : this.matrix) {
            for (double val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }

        System.out.println("---------------------------");
    }

    boolean isEqual(Matrix m) {
        if (m.getColumnCount() != this.getColumnCount() || m.getRowCount() != this.getRowCount()) {
            return false;
        }

        for (int i = 0; i < m.getRowCount(); i++) {
            for (int j = 0; j < m.getColumnCount(); j++) {
                if (m.getElement(i, j) != this.getElement(i, j)) {
                    return false;
                }
            }
        }

        return true;
    }

    public Matrix getStripe(int startRow, int endRow) {
        Matrix stripe = new Matrix(endRow - startRow, this.getColumnCount());

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < this.getColumnCount(); j++) {
                stripe.setElement(i - startRow, j, this.getElement(i, j));
            }
        }

        return stripe;
    }

    public double[] convertToArray1D() {
        double[] arr1D = new double[rowNum * colNum];
        int index = 0;

        for (int i = 0; i < this.rowNum; i++) {
            for (int j = 0; j < this.colNum; j++) {
                arr1D[index] = this.matrix[i][j];
                index++;
            }
        }

        return arr1D;
    }
}
