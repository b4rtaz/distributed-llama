long timeMs();
float randomF32(unsigned long long *state);

void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
