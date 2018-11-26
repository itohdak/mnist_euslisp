(defvar *eus-cuda-matrix-library*
  (load-foreign "./eus_cuda_matrix/build/libeus_cuda_matrix.so"))
;; (defvar *cblas-library*
;;   (load-foreign (format nil "/usr/lib/libcblas.so")))
;; (defvar *openblas-library*
;;   (load-foreign "/usr/lib/openblas-base/libblas.so"))


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
(defforeign _call-vectorAddCUBLAS
  *eus-cuda-matrix-library*
  "call_vectorAddCUBLAS"
  ()
  :integer
  )

;;  ( CBLAS_LAYOUT CBLAS_TRANSPOSE CBLAS_TRANSPOSE :integer :integer :integer :float :integer :integer :integer :integer :float :integer :integer)
;; (defforeign cblas_dgemm
;;   *cblas-library*
;;   "cblas_dgemm"
;;   ()
;;   :integer
;;   )

;; (defforeign openblas_dgemm
;;   *openblas-library*
;;   "cblas_dgemm"
;;   ()
;;   :integer
;;   )

(defun fill-zero (a b)
  (let* ((a-row (array-dimension a 0))
	 (a-col (array-dimension a 1))
	 (b-row (array-dimension b 0))
	 (b-col (array-dimension b 1))
	 (start1 0)
	 (start2 0)
	 (end2 b-col))
    (dotimes (i a-row)
      (replace (a . entity) (b . entity)
	       :start1 start1
	       :start2 start2 :end2 end2)
      (setq start1 (+ start1 a-col) start2 (+ start2 b-col) end2 (+ end2 b-col))))
  a)

(defun unfill-zero (a b)
  (let* ((a-row (array-dimension a 0))
	 (a-col (array-dimension a 1))
	 (b-row (array-dimension b 0))
	 (b-col (array-dimension b 1))
	 (start1 0)
	 (start2 0)
	 (end2 a-col))
    (dotimes (i b-row)
      (replace (b . entity) (a . entity)
	       :start1 start1
	       :start2 start2 :end2 end2)
      (setq start1 (+ start1 b-col) start2 (+ start2 a-col) end2 (+ end2 a-col))))
  b)

(defun cuda-dgemm
  (a b c)
  (let* ((row-a (* (ceiling (/ (array-dimension a 0) 32.0)) 32)) ;; 32
         (col-a (* (ceiling (/ (array-dimension a 1) 32.0)) 32))
         (col-b (* (ceiling (/ (array-dimension b 1) 32.0)) 32))
         (mat-a
          (fill-zero
	   (make-array (list row-a col-a) :element-type :float)
	   a))
         (mat-b
          (fill-zero
	   (make-array (list col-a col-b) :element-type :float)
	   b))
         (mat-c
          (fill-zero
	   (make-array (list row-a col-b) :element-type :float)
	   c))
         )

    (_call-matrixMultiply row-a col-a col-b (mat-a . entity) (mat-b . entity) (mat-c . entity))
    (unfill-zero mat-c c)
    ))

(defun cuda-cublas-m* (a b c)
  (let ((row-a (array-dimension a 0))
        (col-a (array-dimension a 1))
        (col-b (array-dimension b 1))
        )
    (_call-matrixMultiplyCUBLAS
     row-a col-a col-b
     (a . entity)
     (b . entity)
     (c . entity)
     )
    c)
  )

(defun cuda-cublas-v+ (a b &key (alpha 1.0))
  (let ((n (length a))
        )
    (_call-vectorAddCUBLAS
     n
     alpha a
     b
     )
    b)
  )
