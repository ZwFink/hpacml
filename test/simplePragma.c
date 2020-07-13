#include <stdio.h>
#include <stdlib.h>

int printHello(){
    printf("Hello World\n");
    return 1;
}

int main(int argc, char *argv[]){
#pragma approx dinos 
    printHello();
}
