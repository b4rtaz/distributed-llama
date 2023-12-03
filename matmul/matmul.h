
//     weights      input    output
//   ___________     ___      ___
//   |         |     | |      | |
// d |         | *   | |  = d | |
//   |_________|   n | |      |_|
//        n          |_|       1
//                    1

void matmul(float* output, float* input, float* weights, int n, int d) {
    // weights (d,n) @ input (n,1) -> output (d,1)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += weights[i * n + j] * input[j];
        }
        output[i] = val;
    }
}
