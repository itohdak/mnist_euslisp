(load "cblaslib.l")
(load "mnist.so")

;; matrix
;; (make-array '(10 10) :element-type :float)

(setq *lr* 0.003) ;; learning rate
(setq *mr* 0.5) ;; momentum rate

(defclass ReLU
  :super propertied-object
  :slots ())
(defmethod ReLU
  (:init ())
  (:call
   (x)
   (let* ((y (copy-object x)))
     (dotimes (i (length y))
       (if (< (elt x i) 0)
	   (setf (elt y i) 0)))
     y))
  (:diff
   (x)
   (let* ((y (copy-object x)))
     (dotimes (i (length y))
       (if (< (elt x i) 0)
	   (setf (elt y i) 0)
	 (setf (elt y i) 1)))
     y))
  )

(defclass Sigmoid
  :super propertied-object
  :slots ())
(defmethod Sigmoid
  (:init ())
  (:call
   (x)
   (let* ((y (copy-object x)))
     (dotimes (i (length y))
       (setf (elt y i) (/ 1.0 (+ 1.0 (exp (- (elt x i)))))))
     y))
  (:diff
   (x)
   (let* ((y (copy-object x))
	  (c (send self :call x)))
     (dotimes (i (length y))
       (setf (elt y i) (* (elt c i) (- 1.0 (elt c i)))))
     y))
  )

(defclass Softmax
  :super propertied-object
  :slots ())
(defmethod Softmax
  (:init ())
  (:call
   (x)
   (let* ((sum 0)
	  (y (copy-object x)))
     (dotimes (i (length x))
       (setq sum (+ sum (exp (elt x i)))))
     (dotimes (i (length y))
       (setf (elt y i) (/ (exp (elt x i)) sum)))
     y))
  (:diff
   (x)
   (let* ((y (copy-object x))
	  (c (send self :call x)))
     (dotimes (i (length y))
       (setf (elt y i) (* (elt c i) (- 1.0 (elt c i)))))
     y))
  )


(defclass Perceptron
  :super propertied-object
  :slots (W b delta activation p mask pre-dW pre-db u z in-dim out-dim))
(defmethod Perceptron
  (:init
   (in-dim- out-dim- p-dropout act)
   (setq in-dim in-dim-)
   (setq out-dim out-dim-)
   (setq W (make-matrix out-dim in-dim))
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
   (setq pre-db nil))
  (:call
   (x)
   "
   W: 2 dim (out-dim x in-dim)
   x: 1 dim (in-dim)
   b: 1 dim (out-dim)

   u: 1 dim (out-dim)
   z: 1 dim (out-dim)
   "
   (setq u (v+ (transform W x) b))
   (setq z (send activation :call u))
   ;; (setq u (cblas-mat+ (cblas-ddot x W) b))
   ;; (setq z (activation u))
   z))

(defclass MultiLayerPerceptron
  :super propertied-object
  :slots (layers y loss))
(defmethod MultiLayerPerceptron
  (:init
   (layers-)
   (setq layers layers-)
   (dolist (layer layers)
     (format t "(in: ~A  out: ~A)~%"
	     (layer . in-dim)
	     (layer . out-dim)))
   (format t "~%")
   self)
  (:train
   (x train-data learning-rate momentum-rate)
   (setq y x)
   (dolist (layer layers)
     ;; (setf (layer . mask)) ;; dropout
     (setq y (send layer :call y))
     ;; (cblas-ddot mask y) ;; dropout
     )
   (setq loss (/ (log (elt y (position-if #'(lambda (x) (= x 1.0)) train-data))) (length x)))

   ;; back propagation
   (setq delta (v- y train-data))
   (setq ((elt layers (1- (length layers))) . delta) delta)
   (setq W ((elt layers (1- (length layers))) . W))

   (dolist (layer (cdr (reverse layers)))
     (let* ((delta-prop (transform (transpose W)
				   (copy-object delta)))
	    (diff (send (layer . activation) :diff (layer . u))))
       (setq delta (make-array (length delta-prop) :element-type :float :initial-element 0.0))
       (dotimes (i (length delta))
	 (setf (elt delta i) (+ (elt delta-prop i) (elt diff i))))
       )
     ;; (setq delta (transform (layer . mask) (copy-object delta)))
     (setq (layer . delta) delta)
     (setq W (layer . W))
     )

   (setq z x)
   (dolist (layer layers)
     ;; (format t "out-dim: ~A~%in-dim : ~A~%z    : ~A~%delta: ~A~%~%"
     ;; 	     (layer . out-dim) (layer . in-dim) (length z) (length (layer . delta)))
     (setq dW (make-matrix (layer . out-dim) (layer . in-dim)))
     (dotimes (i (length (layer . delta)))
       (dotimes (j (length z))
	 (setf (aref dW i j) (* (elt (layer . delta) i) (elt z j)))))

     (setq db (make-array (length (layer . delta)) :element-type :float))
     (dotimes (i (length (layer . delta)))
       (setf (elt db i) (* (elt (layer . delta) i) 1.0)))
     (cblas-daxpy (dW . entity) ((layer . W) . entity) :alpha (- learning-rate))
     ;; (format t "b: ~A~%db: ~A~%~%" (length (layer . b)) (length db))
     (setq (layer . b) (v- (layer . b) (scale learning-rate db)))

     ;; (unless (or (null (layer . pre-dW)) (null (layer . pre-db)))
     ;;   (setq (layer . W) (v+ (layer . W) (scale momentum-rate pre-dW)))
     ;;   (setq (layer . b) (v+ (layer . b) (scale momentum-rate pre-db))))
     ;; (setq (layer . pre-dW) (v- (layer . pre-dW) (scale learning-rate dW)))
     ;; (setq (layer . pre-db) (v- (layer . pre-db) (scale learning-rate db)))
     (setq z (layer . z))
     )

   loss)
  (:print-weight
   ()
   (dolist (layer layers)
     (print ((layer . W) . entity)))
   t)
  )

(defun test-perceptron ()
  (setq p (instance Perceptron :init 10 20 1.0 ReLU))
  (setq x (make-array 10 :element-type :float))
  (send p :call x)
  t)

(defun test-mlp ()
  (setq mlp (instance MultiLayerPerceptron :init
		      (list (instance Perceptron :init 10 20 1.0 ReLU)
			    (instance Perceptron :init 20 20 1.0 ReLU)
			    (instance Perceptron :init 20 10 1.0 Softmax))))
  (setq x (make-array 10 :element-type :float))
  (setq train #f(0 0 0 0 0 1 0 0 0 0))
  (dotimes (i 100)
    (print (send mlp :train x train *lr* *mr*)))
  ;; (send mlp :print-weight)
  t)

(defun test-mnist ()
  (setq mlp (instance MultiLayerPerceptron :init
		      (list (instance Perceptron :init 10 20 1.0 ReLU)
			    (instance Perceptron :init 20 20 1.0 ReLU)
			    (instance Perceptron :init 20 10 1.0 Softmax))))
  (dotimes (i 100)
    (setq x (elt *train-images* i))
    (setq train (elt *train-labels* i))
    (send mlp :train x train *lr* *mr*)
    )
  )
