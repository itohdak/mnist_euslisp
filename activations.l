(defclass ReLU
  :super propertied-object
  :slots ())
(defmethod ReLU
  (:init ())
  (:call
   (x)
   (cond
    ((vectorp x)
     (let* ((y (copy-object x)))
       (dotimes (i (length y))
	 (if (< (elt x i) 0)
	     (setf (elt y i) 0)))
       y))
    (t
     (let* ((y (copy-object x)))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (setf (aref y i j) (max 0 (aref x i j)))))
       y))
    ))
  (:diff
   (x)
   (cond
    ((vectorp x)
     (let* ((y (copy-object x)))
       (dotimes (i (length y))
	 (if (< (elt x i) 0)
	     (setf (elt y i) 0)
	   (setf (elt y i) 1)))
       y))
    (t
     (let* ((y (copy-object x)))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (if (<= (aref x i j) 0)
	       (setf (aref y i j) 0)
	     (setf (aref y i j) 1))))
       y))
    ))
  )

(defclass Sigmoid
  :super propertied-object
  :slots ())
(defmethod Sigmoid
  (:init ())
  (:call
   (x)
   (cond
    ((vectorp x)
     (let* ((y (copy-object x)))
       (dotimes (i (length y))
	 (setf (elt y i) (/ 1.0 (+ 1.0 (exp (- (elt x i)))))))
       y))
    (t
     (let* ((y (copy-object x)))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (setf (aref y i j) (/ 1.0 (+ 1.0 (exp (- (aref x i j))))))))
       y))
    ))
  (:diff
   (x)
   (cond
    ((vectorp x)
     (let* ((y (copy-object x))
	    (c (send self :call x)))
       (dotimes (i (length y))
	 (setf (elt y i) (* (elt c i) (- 1.0 (elt c i)))))
       y))
    (t
     (let* ((y (copy-object x))
	    (c (send self :call x)))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (setf (aref y i j) (* (aref c i j) (- 1.0 (aref c i j))))))
       y))
    ))
  )

(defclass Softmax
  :super propertied-object
  :slots ())
(defmethod Softmax
  (:init ())
  (:call
   (x)
   (cond
    ((vectorp x)
     (let* ((sum 0)
	    (y (copy-object x)))
       (setq sum (reduce #'(lambda (a b)
			     (+ a (exp b)))
			 (concatenate float-vector #f(0) x)))
       (dotimes (i (length y))
	 (setf (elt y i) (/ (exp (elt x i)) sum)))
       y))
    (t
     (let* ((n (array-dimension x 0)) ;; # of data
	    (d (array-dimension x 1)) ;; # of dim
	    (sum (make-array n :element-type :float :initial-element 0.0))
	    (y (copy-object x)))
       (dotimes (i n)
	 (let* ((x- (subseq (x . entity) (* i d) (* (1+ i) d))))
	   (setf (elt sum i)
		 (reduce #'(lambda (a b)
			     (+ a (exp b)))
			 (concatenate float-vector #f(0) x-)))))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (setf (aref y i j) (/ (exp (aref x i j)) (elt sum i)))))
       y))
    ))
  (:diff
   (x)
   (cond
    ((vectorp x)
     (let* ((y (copy-object x))
	    (c (send self :call x)))
       (dotimes (i (length y))
	 (setf (elt y i) (* (elt c i) (- 1.0 (elt c i)))))
       y))
    (t
     (let* ((y (copy-object x))
	    (c (send self :call x)))
       (dotimes (i (array-dimension y 0))
	 (dotimes (j (array-dimension y 1))
	   (setf (aref y i j) (* (aref c i j) (- 1.0 (aref c i j))))))
       y))
    ))
  )
