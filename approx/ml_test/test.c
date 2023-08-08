#include <stdio.h>
#include <approx.h> 
#include <stdlib.h>

#define SIZE (1024)

void test(float *y, float *x, int size, float a, float b, float c){
  for (int i = 0; i < 2*size-1; i+=2){
    float tmp = x[i] * a + b * x[i+1] + c;
    if ( y[i/2] != tmp)
     printf("Error in %d, %f-%f\n", i/2, tmp, y[i/2]);
  }
}


//Linear function : y = x1 * a + x2 * b + c
void foo(float *y, float *x, int size, float a, float b, float c){
  char name[100];
  for (int i = 0; i < 2*size-1; i+=2){
#pragma approx ml(offline) in(x[i:2]) out(y[i/2]) label("test_region")
    y[i/2] = x[i] * a + b * x[i+1] + c;
  }
}

int main(int argc, char *argv[]){
 float *x = (float*) malloc (sizeof(float)*SIZE*2);
 float *y = (float*) malloc (sizeof(float)*SIZE);
 for (int i = 0; i < 2*SIZE; i++)
  x[i] = (float) rand() / (float) RAND_MAX;

 foo(y, x, SIZE, 0.354f, 10.0, 5.45f);
 test(y, x, SIZE, 0.354f, 10.0, 5.45f);
 free(x);
 free(y);
 return 0;
}
