import mpi.*;

public class Main {
    static final int NRA = 1000;
    static final int NCA = 1000;
    static final int NCB = 1000;
    static final int MASTER = 0;
    static final int FROM_MASTER = 1;
    static final int FROM_WORKER = 4;

    public static void main(String[] args) {
        performMatrixMultiplicationNonBlocking(args);
    }

    public static void performMatrixMultiplicationBlocking(String[] args) {
        MPI.Init(args);

        int currentProcess = MPI.COMM_WORLD.Rank();
        int processesCount = MPI.COMM_WORLD.Size();

        int workersCount = processesCount - 1;
        if (NRA % workersCount != 0) {
            if (currentProcess == MASTER) {
                System.out.println("Error!");
            }

            MPI.Finalize();
            return;
        }

        int rowsPerProcess = NRA / workersCount;

        if (currentProcess == MASTER) {
            Matrix matrixA = new Matrix(NRA, NCA);
            Matrix matrixB = new Matrix(NCA, NCB);
            Matrix matrixC = new Matrix(NRA, NCB);

            matrixA.fill(2);
            matrixB.fill(2);

            long startTime = System.nanoTime();

            double[] arrB = matrixB.convertToArray1D();

            for (int dest = 1; dest <= workersCount; dest++) {
                int offset = (dest - 1);
                int startRow = offset * rowsPerProcess;
                int endRow = startRow + rowsPerProcess;

                double[] arrSubA = (matrixA.getStripe(startRow, endRow)).convertToArray1D();
                MPI.COMM_WORLD.Send(new int[]{offset}, 0, 1, MPI.INT, dest, FROM_MASTER);
                MPI.COMM_WORLD.Send(arrSubA, 0, NCA * rowsPerProcess, MPI.DOUBLE, dest, FROM_MASTER + 1);
                MPI.COMM_WORLD.Send(arrB, 0, NCA * NCB, MPI.DOUBLE, dest, FROM_MASTER + 2);
            }

            for (int source = 1; source <= workersCount; source++) {
                int[] offset = new int[1];
                MPI.COMM_WORLD.Recv(offset, 0, 1, MPI.INT, source, FROM_WORKER);

                double[] arrSubC = new double[rowsPerProcess * NCB];
                MPI.COMM_WORLD.Recv(arrSubC, 0, rowsPerProcess * NCB, MPI.DOUBLE, source, FROM_WORKER + 1);
                matrixC.setStripeByArr1D(arrSubC, offset[0] * rowsPerProcess);
            }

            long endTime = System.nanoTime();
            System.out.println("Matrix A dimensions: " + NRA + "x" + NCA);
            System.out.println("Matrix B dimensions: " + NCA + "x" + NCB);
            System.out.println("Matrix C dimensions: " + NRA + "x" + NCB);
            System.out.println("Matrix Multiplication time (MPI one-to-one, blocking): " + (endTime - startTime) / 1000000000.0);

            //matrixC.print();
        }
        else {
            int[] offset = new int[1];
            MPI.COMM_WORLD.Recv(offset, 0, 1, MPI.INT, MASTER, FROM_MASTER);

            double[] arrSubA = new double[rowsPerProcess * NCA];
            MPI.COMM_WORLD.Recv(arrSubA, 0, rowsPerProcess * NCA, MPI.DOUBLE, MASTER, FROM_MASTER + 1);
            Matrix subMatrixA = new Matrix(arrSubA, rowsPerProcess, NCA);

            double[] arrB = new double[NCA * NCB];
            MPI.COMM_WORLD.Recv(arrB, 0, NCA * NCB, MPI.DOUBLE, MASTER, FROM_MASTER + 2);
            Matrix matrixB = new Matrix(arrB, NCA, NCB);

            Matrix matrixSubC = new Matrix(rowsPerProcess, NCB);
            for (int i = 0; i < rowsPerProcess; i++) {
                for (int j = 0; j < NCB; j++) {
                    for (int k = 0; k < NCA; k++) {
                        matrixSubC.setElement(i, j, matrixSubC.getElement(i, j) + subMatrixA.getElement(i, k) * matrixB.getElement(k, j));
                    }
                }
            }

            MPI.COMM_WORLD.Send(offset, 0, 1, MPI.INT, MASTER, FROM_WORKER);
            MPI.COMM_WORLD.Send(matrixSubC.convertToArray1D(), 0, rowsPerProcess * NCB, MPI.DOUBLE, MASTER, FROM_WORKER + 1);
        }

        MPI.Finalize();
    }

    public static void performMatrixMultiplicationNonBlocking(String[] args) {
        MPI.Init(args);

        int currentProcess = MPI.COMM_WORLD.Rank();
        int processesCount = MPI.COMM_WORLD.Size();

        int workersCount = processesCount - 1;
        if (NRA % workersCount != 0) {
            if (currentProcess == MASTER) {
                System.out.println("Error!");
            }

            MPI.Finalize();
            return;
        }

        int rowsPerProcess = NRA / workersCount;

        if (currentProcess == MASTER) {
            Matrix matrixA = new Matrix(NRA, NCA);
            Matrix matrixB = new Matrix(NCA, NCB);
            Matrix matrixC = new Matrix(NRA, NCB);

            matrixA.fill(2);
            matrixB.fill(2);

            long startTime = System.nanoTime();

            double[] arrB = matrixB.convertToArray1D();

            for (int dest = 1; dest <= workersCount; dest++) {
                int offset = (dest - 1);
                int startRow = offset * rowsPerProcess;
                int endRow = startRow + rowsPerProcess;

                double[] arrSubA = (matrixA.getStripe(startRow, endRow)).convertToArray1D();
                var offsetSend = MPI.COMM_WORLD.Isend(new int[]{offset}, 0, 1, MPI.INT, dest, FROM_MASTER);
                var subASend = MPI.COMM_WORLD.Isend(arrSubA, 0, NCA * rowsPerProcess, MPI.DOUBLE, dest, FROM_MASTER + 1);
                var bSend = MPI.COMM_WORLD.Isend(arrB, 0, NCA * NCB, MPI.DOUBLE, dest, FROM_MASTER + 2);
            }

            for (int source = 1; source <= workersCount; source++) {
                int[] offset = new int[1];
                var offsetReceive = MPI.COMM_WORLD.Irecv(offset, 0, 1, MPI.INT, source, FROM_WORKER);

                double[] arrSubC = new double[rowsPerProcess * NCB];
                var subCReceive = MPI.COMM_WORLD.Irecv(arrSubC, 0, rowsPerProcess * NCB, MPI.DOUBLE, source, FROM_WORKER + 1);

                offsetReceive.Wait();
                subCReceive.Wait();

                matrixC.setStripeByArr1D(arrSubC, offset[0] * rowsPerProcess);
            }

            long endTime = System.nanoTime();
            System.out.println("Matrix A dimensions: " + NRA + "x" + NCA);
            System.out.println("Matrix B dimensions: " + NCA + "x" + NCB);
            System.out.println("Matrix C dimensions: " + NRA + "x" + NCB);
            System.out.println("Matrix Multiplication time (MPI one-to-one, non-blocking): " + (endTime - startTime) / 1000000000.0);

            //matrixC.print();
        }
        else {
            int[] offset = new int[1];
            var offsetReceive = MPI.COMM_WORLD.Irecv(offset, 0, 1, MPI.INT, MASTER, FROM_MASTER);

            double[] arrSubA = new double[rowsPerProcess * NCA];
            var subAReceive = MPI.COMM_WORLD.Irecv(arrSubA, 0, rowsPerProcess * NCA, MPI.DOUBLE, MASTER, FROM_MASTER + 1);
            subAReceive.Wait();
            Matrix subMatrixA = new Matrix(arrSubA, rowsPerProcess, NCA);

            double[] arrB = new double[NCA * NCB];
            var bReceive = MPI.COMM_WORLD.Irecv(arrB, 0, NCA * NCB, MPI.DOUBLE, MASTER, FROM_MASTER + 2);
            bReceive.Wait();
            Matrix matrixB = new Matrix(arrB, NCA, NCB);

            Matrix matrixSubC = new Matrix(rowsPerProcess, NCB);
            for (int i = 0; i < rowsPerProcess; i++) {
                for (int j = 0; j < NCB; j++) {
                    for (int k = 0; k < NCA; k++) {
                        matrixSubC.setElement(i, j, matrixSubC.getElement(i, j) + subMatrixA.getElement(i, k) * matrixB.getElement(k, j));
                    }
                }
            }

            offsetReceive.Wait();
            var offsetSend = MPI.COMM_WORLD.Isend(offset, 0, 1, MPI.INT, MASTER, FROM_WORKER);
            var subCSend = MPI.COMM_WORLD.Isend(matrixSubC.convertToArray1D(), 0, rowsPerProcess * NCB, MPI.DOUBLE, MASTER, FROM_WORKER + 1);

        }

        MPI.Finalize();
    }
}