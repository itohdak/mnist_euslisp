(defvar *eus-cuda-matrix-library*
  (load-foreign "../../build/libeus_cuda_matrix.so"))
(defvar *cblas-library*
  (load-foreign (format nil "/usr/lib/libcblas.so")))
(defvar *openblas-library*
  (load-foreign "/usr/lib/openblas-base/libblas.so"))


;; int call_matrixMultiply(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C);
(defforeign _call-matrixMultiply
  *eus-cuda-matrix-library*
  "call_matrixMultiply"
  (
   :integer
   :integer
   :integer
   :string
   :string
   :string
   )
  :integer
  )

;; int call_matrixMultiplyCUBLAS(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C);
(defforeign _call-matrixMultiplyCUBLAS
  *eus-cuda-matrix-library*
  "call_matrixMultiplyCUBLAS"
  (
   :integer
   :integer
   :integer
   :string
   :string
   :string
   )
  :integer
  )

;;  ( CBLAS_LAYOUT CBLAS_TRANSPOSE CBLAS_TRANSPOSE :integer :integer :integer :float :integer :integer :integer :integer :float :integer :integer)
(defforeign cblas_dgemm
  *cblas-library*
  "cblas_dgemm"
  ()
  :integer
  )

(defforeign openblas_dgemm
  *openblas-library*
  "cblas_dgemm"
  ()
  :integer
  )
