;;
(require "cblaslib.l")
;;(require "activations.so")
;; (require "mnist.so")
(eval-when (load eval)
  (load-library "./MATPROD/Linux64/libmatprod" '("matprod"))
  (load "mnist-draw.l")
  )

(setq *lr* 0.001) ;; learning rate
(setq *mr* 0.5) ;; momentum rate

(defun extended-matrix (v n)
  """
  ベクトルvを行にして，n行複製した行列を作る
  """
  (let* ((len (length v))
	 (ret (make-array `(,n ,len) :element-type :float))
	 (lenxn (* len n)))
    (do ((i 0 (+ i len))) ((>= i lenxn) ret)
      (replace (ret . entity) v :start1 i))
    ))

(defun one-matrix (dimensions)
  """
  要素がすべて1で，形がdimensionsの行列を作る
  """
  (make-array dimensions :element-type :float :initial-element 1.0)
  )

(defclass Perceptron
  :super propertied-object
  :slots (W Wt b delta activation p mask pre-dW pre-db u z in-dim out-dim dW db))
(defmethod Perceptron
  (:init
   (in-dim- out-dim- p-dropout act)
   (setq in-dim in-dim-)
   (setq out-dim out-dim-)
   (setq W (make-matrix out-dim in-dim))
   (setq Wt (transpose W))
   (dotimes (i (array-dimension W 0))
     (dotimes (j (array-dimension W 1))
       (setf (aref W i j) (- (random 0.16) 0.08))
       ))
   (setq b (make-array out-dim
		       :element-type :float
		       :initial-element 0.0))
   (setq delta nil)
   ;;(setq activation (instance act :init))
   (setq activation act)
   ;; (setq p p-dropout)
   ;; (setq mask (make-matrix out-dim))
   (setq pre-dW nil)
   (setq pre-db nil)
   (setq dW nil)
   (setq db nil)
   self)
  (:call
   (x)
   "
   W,Wt: 2 dim (out-dim x in-dim)
   x: 1 dim (in-dim)
   b: 1 dim (out-dim)
   u: 1 dim (out-dim)
   z: 1 dim (out-dim)
   "
   (setq u (cblas-dgemm x Wt (extended-matrix b (array-dimension x 0))))
   ;;(setq z (send activation :call u))
   (setq z (funcall (symbol-function activation) 0 u)))
  (:dimensions () `(,out-dim ,in-dim))
  (:b () b)
  (:W () W)
  (:Wt () Wt)
  (:delta (new-delta) (setq delta new-delta))
  )

(defclass MultiLayerPerceptron
  :super propertied-object
  :slots (layers loss))

(defmethod MultiLayerPerceptron
  (:layers nil layers)
  (:loss nil loss)
  (:init
   (layers-)
   (setq layers layers-)
   (dolist (layer layers)
     (format t "(in:  ~4D    out:  ~4D)~%"
	     (layer . in-dim)
	     (layer . out-dim)))
   (format t "~%")
   self)
  (:test
   (x)
   (let* ((y (extended-matrix x 1)))
     (dolist (layer layers)
       (setq y (send layer :call y)))
     y))
  (:train-batch
   (x train-data learning-rate momentum-rate)
   (let* ((y x) (z x)
	  loss delta W
	  (last-layer (car (last layers))))
     ;; (setq y x)
     (dolist (layer layers)
       (setq y (send layer :call y)))

     (let* ((loss-tmp 0.0))
       (dotimes (i (array-dimension x 0))
	 (setq loss-tmp
	       (+ loss-tmp
		  (- (log (aref y i
				(position-if #'(lambda (x) (= x 1.0))
					     (matrix-row train-data i))))))))
       (setq loss (/ loss-tmp (array-dimension x 0)))
       )
     ;; back propagation
     (setq delta (copy-object y))
     (cblas-daxpy (train-data . entity) (delta . entity) :alpha -1.0)
     (send last-layer :delta delta)
     (setq W (send last-layer :W))
     ;; (setq ((elt layers (1- (length layers))) . delta) delta)
     ;; (setq W ((elt layers (1- (length layers))) . W))

     (dolist (layer (cdr (reverse layers)))
       (let* ((new-delta (make-array `(,(array-dimension delta 0) ,(layer . out-dim))
				     :element-type :float)))
	 (cblas-dgemm delta W new-delta) ;; 行列積
	 (setq (new-delta . entity)
	       ((mprod new-delta 
		       (funcall (symbol-function (layer . activation))
				1 (layer . u))) . entity)
	       ) ;; 要素積
	 (setq delta new-delta)
	 ;; TODO: dropout
	 (setq (layer . delta) delta)
	 (setq W (layer . W))
	 ))

     ;; update weight
     ;; (setq z x)
     (dolist (layer layers)
       (let* ((dW (make-array (array-dimensions (layer . W)) :element-type :float))
	      (db (make-array `(1 ,(length (layer . b))) :element-type :float)))
	 (cblas-dgemm (transpose (layer . delta)) z dW)
	 (cblas-dgemm (one-matrix `(1 ,(array-dimension z 0)))
		      (layer . delta) db)
	 (cblas-daxpy (dW . entity) ((layer . W) . entity) :alpha (- learning-rate))
	 (setq (layer . Wt) (transpose (layer . W)))
	 (setq (layer . b) (v- (layer . b) (scale learning-rate (db . entity))))
	 ;; TODO: momentum
	 (setq z (layer . z))
	 ))

     ;; (format t "W: ~A~%" (subseq (((elt layers 0) . W) . entity) 0 5))
     ;; (format t "b: ~A~%" (subseq ((elt layers 0) . b) 0 5))

     loss))
  (:print-weight
   ()
   (dolist (layer layers)
     (print ((layer . W) . entity)))
   t)
  )

(defun test-mnist-train (&optional (i 0) &aux y (j 0))
  (unless (boundp '*mlp*)
    (format t ";;loading mnist-mlp-19.l~%")
    (load "mnist-mlp-19.l")
    (format t ";;loaded mnist-mlp-19.l~%")
    )
  (unless (boundp '*train-images*)
    (format t ";;loading mnist-datasets.l~%")
    (load "mnist-datasets.l")
    (format t ";;loaded mnist-datasets.l~%")
    )
  (catch :exit-train
    (do-until-key
     (setq y (send *mlp* :test (elt *train-images* i)))
     (if (null (draw-test-image i j y *train-images* *train-labels*))
	 (incf j))
     (incf i)
     (if (>= i (length *train-images*)) (throw :exit-train nil))))
  )

(defun test-mnist-test (&optional (i 0) &aux y (j 0))
  (unless (boundp '*mlp*)
    (format t ";;loading mnist-mlp-19.l~%")
    (load "mnist-mlp-19.l")
    (format t ";;loaded mnist-mlp-19.l~%")
    )
  (unless (boundp '*test-images*)
    (format t ";;loading mnist-datasets.l~%")
    (load "mnist-datasets.l")
    (format t ";;loaded mnist-datasets.l~%")
    )
  (catch :exit-test
    (do-until-key
     (setq y (send *mlp* :test (elt *test-images* i)))
     (if (null (draw-test-image i j y *test-images* *test-labels*))
	 (incf j))
     (incf i)
     (if (>= i (length *test-images*)) (throw :exit-test nil))))
  )

(defun test-mnist-batch (&optional (batchsize 50))
  (if (>= batchsize 50) (sys:alloc 100000000))
  (unless (boundp '*train-images*)
    (format t "Loading datasets ... mnist-datasets.l~%")
    (require "mnist-datasets.l")
    (format t "Loaded datasets ...~%")
    )
  (setq mlp
	(instance MultiLayerPerceptron :init
		  (list (instance Perceptron :init 784 1000 1.0 'mReLU)
			(instance Perceptron :init 1000 1000 1.0 'mReLU)
			(instance Perceptron :init 1000 10 1.0 'mSoftmax))))
  (format t "learning rate:  ~A~%" *lr*)
  (let* ((tstart))
    (dotimes (epoch 20)
      (format t "epoch:  ~2D  ===========================~%" (1+ epoch))
      (setq tstart (unix::runtime))
      (let* ((loss 0.0)
	     (n 0)
	     (n-batch (/ (length *train-images*) batchsize))
	     )
	(dotimes (i n-batch)
	  (let* ((x (make-array `(,batchsize 784) :element-type :float))
		 (train (make-array `(,batchsize 10) :element-type :float))
		 )
	    ;; create batch
	    (let* ((start (* i batchsize)))
	      (dotimes (j batchsize)
		(replace (x . entity) (elt *train-images* (+ start j))
			 :start1 (* j 784))
		(let* ((label (elt *train-labels* (+ start j)))
		       (ind (ceiling (elt label 0))))
		  (setf (aref train j ind) 1.0))
		))
	    ;; train
	    (let* ((ratio (/ (* 1.0 n) (+ n (array-dimension x 0))))
		   (loss-tmp (send mlp :train-batch x train *lr* *mr*)))
	      (setq loss
		    (+ (* ratio loss)
		       (* (- 1.0 ratio) loss-tmp)))
	      (setq n (+ n (array-dimension x 0)))
	      (format
	       t
	       "#image:  ~5D      loss ave.:  ~2,4F (n:  ~5D)      loss:  ~2,4F~%"
	       (* (1+ i) batchsize) loss n loss-tmp)
	      )
	    ))
	(format t "time:  ~S     loss ave.:  ~2,4F~%"
		(* (/ 1000.0 internal-time-units-per-second)
		   (- (unix::runtime) tstart)) loss)
	(setq *mlp* mlp)
	(dump-mnist-instance (format nil "mnist-mlp-~A.l" epoch))
	)
      )
    )
  )

(format t ";;(test-mnist-batch 200) ;; train from train-images~%")
(format t ";;(test-mnist-test) ;; test test-images~%")
(format t ";;(test-mnist-train) ;; test train-images~%")
