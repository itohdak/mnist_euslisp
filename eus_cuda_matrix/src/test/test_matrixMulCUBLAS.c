#include <stdlib.h>
#include <stdio.h>

int call_matrixMultiplyCUBLAS(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C);

void print_matrix(int row, int col, double *matrix)
{
  int i, j;
  for (i = 0; i < row; i++) {
    for (j = 0; j < col; j++) {
      printf("%5.1f", matrix[i * col + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  int row_A = 32;
  int col_A = 32;
  int col_B = 64;
  int size_A = row_A * col_A;
  int size_B = col_A * col_B;
  int size_C = row_A * col_B;
  unsigned int mem_size_A = sizeof(double) * size_A;
  unsigned int mem_size_B = sizeof(double) * size_B;
  unsigned int mem_size_C = sizeof(double) * size_C;
  double *A = (double *)malloc(mem_size_A);
  double *B = (double *)malloc(mem_size_B);
  double *C = (double *)malloc(mem_size_C);

  int i;
  for (i = 0; i < size_A; i++) {
    A[i] = 1.0;
  }
  for (i = 0; i < size_B; i++) {
    B[i] = 1.0;
  }

  call_matrixMultiplyCUBLAS(row_A, col_A, col_B, A, B, C);

  printf("matrix A =\n");
  print_matrix(row_A, col_A, A);
  printf("matrix B =\n");
  print_matrix(col_A, col_B, B);
  printf("matrix C =\n");
  print_matrix(row_A, col_B, C);

  free(A);
  free(B);
  free(C);

  return 0;
}
