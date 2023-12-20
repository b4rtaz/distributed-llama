#ifndef funcs_hpp
#define funcs_hpp

long timeMs();
float randomF32(unsigned long long *state);

void rmsnorm(float* o, const float* x, const float* weight, const int size);
void softmax(float* x, const int size);

#endif