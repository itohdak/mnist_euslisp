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
