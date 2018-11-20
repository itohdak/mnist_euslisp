;;
;;
;;
(defun mnist-read-a-2d-image
  (f width height
     &optional (img (make-array (list width height)
				:element-type :byte)))
  (catch :mnist-exit
    (let ((r nil) v)
      (dotimes (i height)
	(dotimes (j width)
	  (if (eq (setq v (read f nil r)) r)
	      (throw :mnist-exit nil)
	    (setf (aref img i j) v)))))
    img))

(defun mnist-read-2d-image
  (fname width height)
  (let (res img eof)
    (with-open-file
     (f fname :direction :input)
     (while (setq img (mnist-read-a-2d-image f width height))
       (push img res)))
    res))

;; (defun mnist-read-a-image
;;   (f length
;;      &optional (n 1.0) (img (make-array length :element-type :byte)))
;;   (catch :mnist-exit
;;     (let ((r '(nil)) v)
;;       (dotimes (i length)
;; 	(if (eq (setq v (read f nil r)) r)
;; 	    (throw :mnist-exit nil)
;; 	  (setf (aref img i) v))))
;;     (scale n (coerce img float-vector))))
(defun mnist-read-a-image
    (f length
       &optional (n 1.0) (img (make-array length :element-type :float)))
  (catch :mnist-exit
    (let ((r '(nil)) v)
      (dotimes (i length)
	(if (eq (setq v (read f nil r)) r)
	    (throw :mnist-exit nil)
	  (setf (aref img i) (* v n)))))
    img))

(defun mnist-read-image
  (fname length
	 &optional (n 1.0))
  (let (res img eof)
    (with-open-file
     (f fname :direction :input)
     (while (setq img (mnist-read-a-image f length n))
       (push img res)))
    res))


(defun test-2d nil
  (setq *test-images* (mnist-read-2d-image "test-images.txt" 28 28))
  (format t "test-images:~A~%" (length *test-images*))
  (setq *test-labels* (mnist-read-2d-image "test-labels.txt" 1 1))
  (format t "test-labels:~A~%" (length *test-labels*))
  (setq *train-labels* (mnist-read-2d-image "train-labels.txt" 1 1))
  (format t "train-labels:~A~%" (length *train-labels*))
  (setq *train-images* (mnist-read-2d-image "train-images.txt" 28 28))
  (format t "train-images:~A~%" (length *train-images*))
  )

(defun test nil
  (let* ((start (unix::runtime)))
    (setq *test-images* (mnist-read-image "test-images.txt" 784 (/ 1.0 255.0)))
    (format t "test-images:~A~%" (length *test-images*))
    (format t "load time: ~A~%" (* (/ 1000.0 internal-time-units-per-second) (- (unix::runtime) start)))
    (setq start (unix::runtime))
    (setq *test-labels* (mnist-read-image "test-labels.txt" 1 1.0))
    (format t "test-labels:~A~%" (length *test-labels*))
    (format t "load time: ~A~%" (* (/ 1000.0 internal-time-units-per-second) (- (unix::runtime) start)))
    (setq start (unix::runtime))
    (setq *train-labels* (mnist-read-image "train-labels.txt" 1 1.0))
    (format t "train-labels:~A~%" (length *train-labels*))
    (format t "load time: ~A~%" (* (/ 1000.0 internal-time-units-per-second) (- (unix::runtime) start)))
    (setq start (unix::runtime))
    (setq *train-images* (mnist-read-image "train-images.txt" 784 (/ 1.0 255.0)))
    (format t "train-images:~A~%" (length *train-images*))
    (format t "load time: ~A~%" (* (/ 1000.0 internal-time-units-per-second) (- (unix::runtime) start)))
    ))


