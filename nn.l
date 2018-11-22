(require "cblaslib.l")
(require "activations.so")
;; (require "mnist.so")

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
   (setq activation (instance act :init))
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
   W: 2 dim (out-dim x in-dim)
   x: 1 dim (in-dim)
   b: 1 dim (out-dim)

   u: 1 dim (out-dim)
   z: 1 dim (out-dim)
   "
   (cond
    ((vectorp x)
     (setq u (v+ (transform W x) b))
     (setq z (send activation :call u)))
    (t
     (let* ((bmat (extended-matrix b (array-dimension x 0))))
       (setq u (cblas-dgemm x Wt;; (transpose W)
			    bmat))
       (setq z (send activation :call u))))
    )
   z)
  (:dimensions () `(,out-dim ,in-dim))
  (:W () W)
  (:delta (new-delta) (setq delta new-delta))
  )

(defclass MultiLayerPerceptron
  :super propertied-object
  :slots (layers y loss))
(defmethod MultiLayerPerceptron
  (:init
   (layers-)
   (setq layers layers-)
   (dolist (layer layers)
     (format t "(in:  ~4D    out:  ~4D)~%"
	     (layer . in-dim)
	     (layer . out-dim)))
   (format t "~%")
   self)
  ;; (:train-one
  ;;  (x train-data learning-rate momentum-rate)
  ;;  (setq y x)

  ;;  (dolist (layer layers)
  ;;    ;; (setf (layer . mask)) ;; dropout
  ;;    (setq y (send layer :call y))
  ;;    ;; (cblas-ddot mask y) ;; dropout
  ;;    )

  ;;  (setq loss (/ (- (log (elt y (position-if #'(lambda (x) (= x 1.0)) train-data)))) 1))

  ;;  ;; back propagation
  ;;  (setq delta (v- y train-data))
  ;;  (setq ((elt layers (1- (length layers))) . delta) delta)
  ;;  (setq W ((elt layers (1- (length layers))) . W))

  ;;  (dolist (layer (cdr (reverse layers)))
  ;;    (let* ((delta-prop (transform (transpose W)
  ;; 				   (copy-object delta)))
  ;; 	    (diff (send (layer . activation) :diff (layer . u))))
  ;;      (setq delta (make-array (length delta-prop) :element-type :float :initial-element 0.0))
  ;;      (dotimes (i (length delta))
  ;; 	 (setf (elt delta i) (+ (elt delta-prop i) (elt diff i))))
  ;;      )
  ;;    ;; (setq delta (transform (layer . mask) (copy-object delta)))
  ;;    ;; (format t "delta: ~A~%" (subseq (layer . delta) 0 5))
  ;;    (setq (layer . delta) delta)
  ;;    (setq W (layer . W))
  ;;    )

  ;;  ;; update weight
  ;;  (setq z x)
  ;;  (dolist (layer layers)
  ;;    (setq dW (make-matrix (layer . out-dim) (layer . in-dim)))
  ;;    (dotimes (i (length (layer . delta)))
  ;;      (dotimes (j (length z))
  ;; 	 (setf (aref dW i j) (* (elt (layer . delta) i) (elt z j)))))

  ;;    (setq db (make-array (layer . out-dim) :element-type :float))
  ;;    (dotimes (i (length (layer . delta)))
  ;;      (setf (elt db i) (* (elt (layer . delta) i) 1.0)))

  ;;    ;; (format t "dW: ~A~%" (subseq (dW . entity) 0 5))
  ;;    ;; (dotimes (a (layer . out-dim))
  ;;    ;;   (dotimes (b (layer . in-dim))
  ;;    ;; 	 (setf (aref (layer . W) a b)
  ;;    ;; 	       (- (aref (layer . W) a b) (* (- learning-rate) (aref dW a b))))
  ;;    ;; 	 ))
  ;;    (cblas-daxpy (dW . entity) ((layer . W) . entity) :alpha (- learning-rate))
  ;;    (setq (layer . b) (v- (layer . b) (scale learning-rate db)))

  ;;    ;; (unless (or (null (layer . pre-dW)) (null (layer . pre-db)))
  ;;    ;;   (setq (layer . W) (v+ (layer . W) (scale momentum-rate pre-dW)))
  ;;    ;;   (setq (layer . b) (v+ (layer . b) (scale momentum-rate pre-db))))
  ;;    ;; (setq (layer . pre-dW) (v- (layer . pre-dW) (scale learning-rate dW)))
  ;;    ;; (setq (layer . pre-db) (v- (layer . pre-db) (scale learning-rate db)))
  ;;    (setq z (copy-object (layer . z)))
  ;;    )

  ;;  ;; (format t "W: ~A~%" (subseq (((elt layers 0) . W) . entity) 0 5))
  ;;  ;; (format t "b: ~A~%" (subseq ((elt layers 0) . b) 0 5))

  ;;  loss)
  ;; (:train
  ;;  (x train-data learning-rate momentum-rate)

  ;;  ;; atom -> list
  ;;  (if (atom x)
  ;;      (setq x (list x)))
  ;;  (if (atom train-data)
  ;;      (setq train-data (list train-data)))

  ;;  (setq pred (mapcar
  ;; 	       #'(lambda (x-)
  ;; 		   (setq y x-)
  ;; 		   (dolist (layer layers)
  ;; 		     (setq y (send layer :call y)))
  ;; 		   y) x))

  ;;  (setq loss
  ;; 	 (/ (reduce #'+ (mapcar
  ;; 			 #'(lambda (p train)
  ;; 			     (/ (- (log (elt p (position-if #'(lambda (x) (= x 1.0)) train)))) 1))
  ;; 			 pred train-data))
  ;; 	    (length x)))

  ;;  (setq delta-all (mapcar
  ;; 		    #'(lambda
  ;; 			(p train)
  ;; 			(v- p train))
  ;; 		    pred train-data))

  ;;  (dolist (layer layers)
  ;;    (setq (layer . dW) (make-matrix (layer . out-dim) (layer . in-dim)))
  ;;    (setq (layer . db) (make-array (layer . out-dim) :element-type :float)))

  ;;  (dotimes (n (length delta-all))
  ;;    (let* ((delta (elt delta-all n))
  ;; 	    (z (elt x n)))
  ;;      ;; back propagation
  ;;      (setq ((elt layers (1- (length layers))) . delta) delta)
  ;;      (setq W ((elt layers (1- (length layers))) . W))

  ;;      (dolist (layer (cdr (reverse layers)))
  ;; 	 (let* ((delta-prop (transform (transpose W)
  ;; 				       (copy-object delta)))
  ;; 		(diff (send (layer . activation) :diff (layer . u))))
  ;; 	   (setq delta (make-array (length delta-prop) :element-type :float :initial-element 0.0))
  ;; 	   (dotimes (i (length delta))
  ;; 	     (setf (elt delta i) (+ (elt delta-prop i) (elt diff i))))
  ;; 	   )
  ;; 	 (setq (layer . delta) delta)
  ;; 	 (setq W (layer . W))
  ;; 	 )

  ;;      ;; update weight
  ;;      (dolist (layer layers)
  ;; 	 (dotimes (i (length (layer . delta)))
  ;; 	   (dotimes (j (length z))
  ;; 	     (setf (aref (layer . dW) i j) (+ (aref (layer . dW) i j) (* (elt (layer . delta) i) (elt z j))))))

  ;; 	 (dotimes (i (length (layer . delta)))
  ;; 	   (setf (elt (layer . db) i) (+ (elt (layer . db) i) (* (elt (layer . delta) i) 1.0))))
  ;; 	 ;; (cblas-daxpy (dW . entity) ((layer . W) . entity) :alpha (- learning-rate))
  ;; 	 ;; (setq (layer . b) (v- (layer . b) (scale learning-rate db)))
  ;; 	 (setq z (copy-object (layer . z)))
  ;; 	 )
  ;;      ))

  ;;  (dolist (layer layers)
  ;;    (cblas-daxpy ((layer . dW) . entity) ((layer . W) . entity) :alpha (- learning-rate))
  ;;    (setq (layer . b) (v- (layer . b) (scale learning-rate (layer . db))))
  ;;    )

  ;;  ;; (print (subseq (((elt layers 0) . W) . entity) (* 10 10) (+ (* 10 10) 10)))

  ;;  loss)
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
		  (- (log (aref y i (position-if #'(lambda (x) (= x 1.0)) (matrix-row train-data i))))))))
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
       (let* ((new-delta (make-array `(,(array-dimension delta 0) ,(layer . in-dim))
				     :element-type :float)))
	 (cblas-dgemm delta W new-delta) ;; 行列積
	 (setq (new-delta . entity) (coerce (mapcar #'* (coerce (new-delta . entity) cons) (coerce ((send (layer . activation) :diff (layer . u)) . entity) cons)) float-vector)) ;; 要素積
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

;; (defun print-info (x name)
;;   (format t "~A~%~A~%~%" name x))

;; (defun test-perceptron ()
;;   (setq p (instance Perceptron :init 10 20 1.0 ReLU))
;;   (setq x (make-array 10 :element-type :float))
;;   (send p :call x)
;;   t)
;; (defun test-perceptron-mat ()
;;   (setq p (instance Perceptron :init 10 20 1.0 ReLU))
;;   (setq x (make-array '(2 10) :element-type :float))
;;   (send p :call x)
;;   t)

;; (defun test-mlp ()
;;   (setq mlp (instance MultiLayerPerceptron :init
;; 		      (list (instance Perceptron :init 10 20 1.0 ReLU)
;; 			    (instance Perceptron :init 20 20 1.0 ReLU)
;; 			    (instance Perceptron :init 20 10 1.0 Softmax))))
;;   (setq x (make-array 10 :element-type :float))
;;   (setq train #f(0 0 0 0 0 1 0 0 0 0))
;;   (dotimes (i 100)
;;     (print (send mlp :train x train *lr* *mr*)))
;;   ;; (send mlp :print-weight)
;;   t)
;; (defun test-mlp-batch ()
;;   (setq mlp (instance MultiLayerPerceptron :init
;; 		      (list (instance Perceptron :init 10 20 1.0 ReLU)
;; 			    (instance Perceptron :init 20 20 1.0 ReLU)
;; 			    (instance Perceptron :init 20 10 1.0 Softmax))))
;;   (setq x (make-array '(10 10) :element-type :float))
;;   (setq train (make-array '(10 10) :element-type :float))
;;   (dotimes (i 10)
;;     (setf (aref train i 0) 1.0))
;;   (dotimes (i 100)
;;     (print (send mlp :train-batch x train *lr* *mr*)))
;;   ;; (send mlp :print-weight)
;;   t)

;; (defun test1 ()
;;   (setq mlp (instance MultiLayerPerceptron :init
;; 		      (list (instance Perceptron :init 50 100 0.5 ReLU)
;; 			    (instance Perceptron :init 100 100 0.5 ReLU)
;; 			    (instance Perceptron :init 100 10 1.0 Softmax))))
;;   (setq x (make-array 50 :element-type :float))
;;   (dotimes (i 50)
;;     (setf (elt x i) i))
;;   (setq train (make-array 10 :element-type :float))
;;   (setf (elt train 0) 1.0)
;;   (setq loss (send mlp :train x train *lr* 0.0))
;;   )

;; (defun test-mnist-one ()
;;   (setq mlp (instance MultiLayerPerceptron :init
;; 		      (list (instance Perceptron :init 784 1000 1.0 ReLU)
;; 			    (instance Perceptron :init 1000 1000 1.0 ReLU)
;; 			    (instance Perceptron :init 1000 10 1.0 Softmax))))

;;   (format t "learning rate: ~A~%" *lr*)
;;   (dotimes (epoc 20)
;;     (format t "=============================================~%epoc: ~A~%~%" (1+ epoc))
;;     (dotimes (i (length *train-images*))
;;       (let* ((loss 0.0))
;; 	(setq x (elt *train-images* i))
;; 	(setq train (make-array 10 :element-type :float :initial-element 0))
;; 	(setq ind (ceiling (elt (elt *train-labels* i) 0)))
;; 	(setf (elt train ind) 1.0)
;; 	(setq loss (+ loss (send mlp :train-one x train *lr* *mr*)))
;; 	(when (= (mod i 100) 99)
;; 	  (format t "~A: ~A~%" (1+ i) (/ loss 100.0))
;; 	  (setq loss 0.0))
;; 	))
;;     )
;;   )
;; (defun test-mnist ()
;;   (setq mlp (instance MultiLayerPerceptron :init
;; 		      (list (instance Perceptron :init 784 1000 1.0 ReLU)
;; 			    (instance Perceptron :init 1000 1000 1.0 ReLU)
;; 			    (instance Perceptron :init 1000 10 1.0 Softmax))))
;;   (setq batchsize 100)
;;   (dotimes (i 100)
;;     (setq x (subseq *train-images* (* i batchsize) (* (1+ i) batchsize)))
;;     (setq train (make-list batchsize :initial-element (make-array 10 :element-type :float :initial-element 0)))
;;     (dotimes (j (length train))
;;       (let* ((label (elt (subseq *train-labels* (* i batchsize) (* (1+ i) batchsize)) j))
;; 	     (ind (ceiling (elt label 0))))
;; 	(setf (elt (elt train j) ind) 1.0))
;;       )
;;     (setq loss (send mlp :train x train *lr* *mr*))
;;     (print loss)
;;     ))
(defun test-mnist-batch (&optional (batchsize 50))
  (if (>= batchsize 50) (sys:alloc 100000000))
  (unless (boundp '*train-images*)
    (format t "Loading datasets ...~%")
    (require "mnist-datasets.l"))
  (setq mlp (instance MultiLayerPerceptron :init
		      (list (instance Perceptron :init 784 1000 1.0 ReLU)
			    (instance Perceptron :init 1000 1000 1.0 ReLU)
			    (instance Perceptron :init 1000 10 1.0 Softmax))))
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
	      (format t "#image:  ~5D      loss ave.:  ~2,4F (n:  ~5D)      loss:  ~2,4F~%" (* (1+ i) batchsize) loss n loss-tmp)
	      )
	    ))
	(format t "time:  ~S     loss ave.:  ~2,4F~%" (* (/ 1000.0 internal-time-units-per-second) (- (unix::runtime) tstart)) loss)
	)
      )
    )
  )

;; (defun draw (image-float-vector)
;;   (unless (boundp '*irtviewer*)
;;     (make-irtviewer)
;;     (send *irtviewer* :change-background #f(1 1 1)))
;;   (let* ((simage (instance color-image :init 28 28)))
;;     (send *irtviewer* :viewer :viewsurface :putimage simage :depth 24)
;;     ))
